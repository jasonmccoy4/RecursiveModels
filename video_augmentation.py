"""
Video Data Augmentation for Something-Something V2

Implements augmentation strategies optimized for SSv2 which has direction-sensitive
labels (e.g., "pushing left to right" vs "pushing right to left").

Key design decisions:
- NO horizontal flip (breaks direction-sensitive labels)
- Temporal augmentations (jittering, variable sampling)
- Spatial augmentations (multi-scale crop, color jitter)
- RandomErasing for regularization

Based on best practices from V-JEPA 2, VideoMAE, and action recognition research.
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Optional, Tuple, List
from dataclasses import dataclass
import pydantic


class VideoAugmentConfig(pydantic.BaseModel):
    """Configuration for video augmentation pipeline."""

    # Temporal augmentation
    temporal_jitter: bool = True  # Random offset in temporal sampling
    temporal_jitter_range: float = 0.1  # Max jitter as fraction of video length

    # Spatial augmentation - RandomResizedCrop
    random_resized_crop: bool = True
    crop_scale_range: Tuple[float, float] = (0.5, 1.0)  # Scale range for cropping
    crop_aspect_ratio_range: Tuple[float, float] = (0.75, 1.333)  # Aspect ratio range

    # Color augmentation (applied consistently across frames)
    color_jitter: bool = True
    brightness: float = 0.4
    contrast: float = 0.4
    saturation: float = 0.4
    hue: float = 0.1
    color_jitter_prob: float = 0.8  # Probability of applying color jitter

    # Grayscale
    random_grayscale: bool = True
    grayscale_prob: float = 0.2

    # Gaussian blur
    gaussian_blur: bool = True
    blur_prob: float = 0.5
    blur_sigma_range: Tuple[float, float] = (0.1, 2.0)

    # Random erasing (cutout-style)
    random_erasing: bool = True
    erasing_prob: float = 0.25
    erasing_scale_range: Tuple[float, float] = (0.02, 0.33)
    erasing_aspect_ratio_range: Tuple[float, float] = (0.3, 3.3)


def get_default_train_augment_config() -> VideoAugmentConfig:
    """Get default augmentation config for SSv2 training."""
    return VideoAugmentConfig()


def get_default_val_augment_config() -> VideoAugmentConfig:
    """Get augmentation config for validation (minimal augmentation)."""
    return VideoAugmentConfig(
        temporal_jitter=False,
        random_resized_crop=False,
        color_jitter=False,
        random_grayscale=False,
        gaussian_blur=False,
        random_erasing=False,
    )


class VideoAugmentor:
    """
    Video augmentation pipeline for Something-Something V2.

    Applies temporally-consistent augmentations that preserve action semantics.
    Explicitly avoids horizontal flip due to direction-sensitive labels.
    """

    def __init__(self, config: VideoAugmentConfig, rng: np.random.Generator):
        self.config = config
        self._rng = rng

    def __call__(self, frames: torch.Tensor, target_size: int) -> torch.Tensor:
        """
        Apply augmentation pipeline to video frames.

        Args:
            frames: Video tensor of shape (T, C, H, W), values in [0, 255]
            target_size: Target spatial size for output

        Returns:
            Augmented frames of shape (T, C, target_size, target_size)
        """
        # Ensure float for processing
        frames = frames.float()

        # 1. Spatial cropping (RandomResizedCrop or center crop)
        if self.config.random_resized_crop:
            frames = self._random_resized_crop(frames, target_size)
        else:
            frames = self._center_crop_resize(frames, target_size)

        # 2. Color jitter (temporally consistent)
        if self.config.color_jitter and self._rng.random() < self.config.color_jitter_prob:
            frames = self._color_jitter(frames)

        # 3. Random grayscale
        if self.config.random_grayscale and self._rng.random() < self.config.grayscale_prob:
            frames = self._to_grayscale(frames)

        # 4. Gaussian blur
        if self.config.gaussian_blur and self._rng.random() < self.config.blur_prob:
            frames = self._gaussian_blur(frames)

        # 5. Random erasing (applied per-frame for diversity)
        if self.config.random_erasing and self._rng.random() < self.config.erasing_prob:
            frames = self._random_erasing(frames)

        return frames

    def _random_resized_crop(self, frames: torch.Tensor, target_size: int) -> torch.Tensor:
        """
        Apply random resized crop consistently across all frames.

        Similar to torchvision.transforms.RandomResizedCrop but for video.
        """
        T, C, H, W = frames.shape

        # Sample crop parameters
        scale = self._rng.uniform(*self.config.crop_scale_range)
        aspect_ratio = self._rng.uniform(*self.config.crop_aspect_ratio_range)

        # Calculate crop size
        area = H * W * scale
        crop_h = int(round(np.sqrt(area / aspect_ratio)))
        crop_w = int(round(np.sqrt(area * aspect_ratio)))

        # Clamp to valid dimensions
        crop_h = min(crop_h, H)
        crop_w = min(crop_w, W)

        # Random crop position
        top = self._rng.integers(0, max(1, H - crop_h + 1))
        left = self._rng.integers(0, max(1, W - crop_w + 1))

        # Crop and resize
        frames = frames[:, :, top:top+crop_h, left:left+crop_w]
        frames = F.interpolate(
            frames,
            size=(target_size, target_size),
            mode='bilinear',
            align_corners=False
        )

        return frames

    def _center_crop_resize(self, frames: torch.Tensor, target_size: int) -> torch.Tensor:
        """Center crop to square and resize."""
        T, C, H, W = frames.shape

        # Center crop to square
        min_dim = min(H, W)
        top = (H - min_dim) // 2
        left = (W - min_dim) // 2
        frames = frames[:, :, top:top+min_dim, left:left+min_dim]

        # Resize
        if frames.shape[-1] != target_size:
            frames = F.interpolate(
                frames,
                size=(target_size, target_size),
                mode='bilinear',
                align_corners=False
            )

        return frames

    def _color_jitter(self, frames: torch.Tensor) -> torch.Tensor:
        """
        Apply color jitter consistently across all frames.

        Adjusts brightness, contrast, saturation, and hue.
        """
        # Normalize to [0, 1] for processing
        frames = frames / 255.0

        # Random order of transforms
        transforms = ['brightness', 'contrast', 'saturation', 'hue']
        self._rng.shuffle(transforms)

        for transform in transforms:
            if transform == 'brightness' and self.config.brightness > 0:
                factor = 1.0 + self._rng.uniform(-self.config.brightness, self.config.brightness)
                frames = frames * factor

            elif transform == 'contrast' and self.config.contrast > 0:
                factor = 1.0 + self._rng.uniform(-self.config.contrast, self.config.contrast)
                mean = frames.mean(dim=(-3, -2, -1), keepdim=True)
                frames = (frames - mean) * factor + mean

            elif transform == 'saturation' and self.config.saturation > 0:
                factor = 1.0 + self._rng.uniform(-self.config.saturation, self.config.saturation)
                # Convert to grayscale for reference
                gray = 0.299 * frames[:, 0:1] + 0.587 * frames[:, 1:2] + 0.114 * frames[:, 2:3]
                frames = frames * factor + gray * (1 - factor)

            elif transform == 'hue' and self.config.hue > 0:
                # Simplified hue rotation via channel cycling blend
                factor = self._rng.uniform(-self.config.hue, self.config.hue)
                if abs(factor) > 0.01:
                    frames = self._adjust_hue(frames, factor)

        # Clamp and convert back to [0, 255]
        frames = torch.clamp(frames, 0, 1) * 255.0
        return frames

    def _adjust_hue(self, frames: torch.Tensor, hue_factor: float) -> torch.Tensor:
        """Adjust hue by rotating in RGB space (simplified approach)."""
        # Convert RGB to HSV-like space
        T, C, H, W = frames.shape

        # Simple hue adjustment via weighted channel mixing
        # This is an approximation but works well for small hue shifts
        angle = hue_factor * np.pi
        cos_a = np.cos(angle)
        sin_a = np.sin(angle)

        # Rotation matrix for hue (simplified)
        r = frames[:, 0:1]
        g = frames[:, 1:2]
        b = frames[:, 2:3]

        # Apply rotation in a luminance-preserving way
        sqrt3 = np.sqrt(3)
        new_r = cos_a * r + (1 - cos_a) / 3 * (r + g + b) + sqrt3 / 3 * sin_a * (g - b)
        new_g = cos_a * g + (1 - cos_a) / 3 * (r + g + b) + sqrt3 / 3 * sin_a * (b - r)
        new_b = cos_a * b + (1 - cos_a) / 3 * (r + g + b) + sqrt3 / 3 * sin_a * (r - g)

        return torch.cat([new_r, new_g, new_b], dim=1)

    def _to_grayscale(self, frames: torch.Tensor) -> torch.Tensor:
        """Convert to grayscale while maintaining 3 channels."""
        # Standard grayscale weights
        gray = 0.299 * frames[:, 0:1] + 0.587 * frames[:, 1:2] + 0.114 * frames[:, 2:3]
        return gray.expand(-1, 3, -1, -1)

    def _gaussian_blur(self, frames: torch.Tensor) -> torch.Tensor:
        """Apply Gaussian blur consistently across all frames."""
        sigma = self._rng.uniform(*self.config.blur_sigma_range)

        # Calculate kernel size (must be odd)
        kernel_size = int(np.ceil(sigma * 3) * 2 + 1)
        kernel_size = max(3, kernel_size)
        if kernel_size % 2 == 0:
            kernel_size += 1

        # Create Gaussian kernel
        x = torch.arange(kernel_size, dtype=frames.dtype, device=frames.device) - kernel_size // 2
        kernel_1d = torch.exp(-x**2 / (2 * sigma**2))
        kernel_1d = kernel_1d / kernel_1d.sum()
        kernel_2d = kernel_1d.view(-1, 1) @ kernel_1d.view(1, -1)
        kernel_2d = kernel_2d.expand(3, 1, -1, -1)

        # Apply blur
        padding = kernel_size // 2
        T, C, H, W = frames.shape

        # Process each frame
        frames_blurred = F.conv2d(
            frames.view(T * C, 1, H, W),
            kernel_2d[0:1],
            padding=padding,
            groups=1
        )
        frames_blurred = frames_blurred.view(T, C, H, W)

        return frames_blurred

    def _random_erasing(self, frames: torch.Tensor) -> torch.Tensor:
        """
        Apply random erasing (cutout) to frames.

        Erases a random rectangle and fills with random values.
        Applied consistently across all frames for temporal coherence.
        """
        T, C, H, W = frames.shape

        # Sample erasing region
        area = H * W
        target_area = area * self._rng.uniform(*self.config.erasing_scale_range)
        aspect_ratio = self._rng.uniform(*self.config.erasing_aspect_ratio_range)

        erase_h = int(round(np.sqrt(target_area / aspect_ratio)))
        erase_w = int(round(np.sqrt(target_area * aspect_ratio)))

        if erase_h < H and erase_w < W:
            top = self._rng.integers(0, H - erase_h)
            left = self._rng.integers(0, W - erase_w)

            # Fill with random values (same across frames for temporal consistency)
            fill_value = torch.from_numpy(
                self._rng.uniform(0, 255, size=(1, C, erase_h, erase_w))
            ).to(frames.dtype).to(frames.device)

            frames[:, :, top:top+erase_h, left:left+erase_w] = fill_value.expand(T, -1, -1, -1)

        return frames


def apply_temporal_jitter(
    total_frames: int,
    num_frames: int,
    rng: np.random.Generator,
    jitter_range: float = 0.1
) -> np.ndarray:
    """
    Sample frame indices with temporal jittering.

    Instead of strictly uniform sampling, adds random offset to each sample point.
    This provides temporal augmentation while preserving action order.

    Args:
        total_frames: Total frames in video
        num_frames: Number of frames to sample
        rng: Random number generator
        jitter_range: Maximum jitter as fraction of segment length

    Returns:
        Array of frame indices to sample
    """
    if total_frames <= num_frames:
        # Not enough frames - use standard linspace with repetition
        return np.linspace(0, total_frames - 1, num_frames, dtype=np.int64)

    # Calculate segment length
    segment_length = total_frames / num_frames

    # Sample from each segment with jitter
    indices = []
    for i in range(num_frames):
        segment_start = i * segment_length
        segment_end = (i + 1) * segment_length
        segment_center = (segment_start + segment_end) / 2

        # Add jitter
        max_jitter = segment_length * jitter_range
        jitter = rng.uniform(-max_jitter, max_jitter)

        # Sample position with jitter, clamped to segment bounds
        sample_pos = segment_center + jitter
        sample_pos = np.clip(sample_pos, segment_start, segment_end - 1)
        sample_pos = np.clip(sample_pos, 0, total_frames - 1)

        indices.append(int(sample_pos))

    return np.array(indices, dtype=np.int64)


def uniform_temporal_sample(total_frames: int, num_frames: int) -> np.ndarray:
    """Standard uniform temporal sampling (for validation)."""
    return np.linspace(0, total_frames - 1, num_frames, dtype=np.int64)
