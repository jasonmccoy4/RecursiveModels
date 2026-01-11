"""
Something-Something V2 Video Dataset

Dataset loader for the Something-Something V2 video classification dataset.
Handles video loading, frame sampling, and batching for V-JEPA 2 training.

Dataset download:
    Download from https://developer.qualcomm.com/software/ai-datasets/something-something
    - 19 WebM video files (~19.4GB total)
    - Annotation JSON files

Expected directory structure:
    data/ssv2/
    ├── videos/                              # Contains all .webm video files
    │   ├── 1.webm
    │   ├── 2.webm
    │   └── ...
    ├── something-something-v2-train.json    # Training annotations
    ├── something-something-v2-validation.json
    └── something-something-v2-labels.json   # 174 class labels
"""

import os
import json
from typing import List, Dict, Optional, Tuple, Iterator
import numpy as np
import torch
from torch.utils.data import IterableDataset, DataLoader
import pydantic

from video_augmentation import (
    VideoAugmentConfig,
    VideoAugmentor,
    apply_temporal_jitter,
    uniform_temporal_sample,
    get_default_train_augment_config,
    get_default_val_augment_config,
)


class SSV2DatasetConfig(pydantic.BaseModel):
    """Configuration for Something-Something V2 dataset."""

    seed: int
    data_root: str  # Path to video files directory
    annotations_path: str  # Path to annotations JSON
    labels_path: str  # Path to labels JSON (174 classes)

    num_frames: int = 64  # V-JEPA 2 expects 64 frames
    frame_size: int = 256  # V-JEPA 2 crop size

    global_batch_size: int
    rank: int = 0
    num_replicas: int = 1

    # Data augmentation config (None = use defaults based on split)
    augment_config: Optional[VideoAugmentConfig] = None


class SSV2DatasetMetadata(pydantic.BaseModel):
    """Metadata for SSv2 dataset."""

    num_classes: int = 174
    num_frames: int = 64
    frame_size: int = 256
    total_samples: int
    sets: List[str] = ["train", "validation"]


def _load_video_torchcodec(video_path: str, frame_indices: np.ndarray) -> torch.Tensor:
    """Load video using torchcodec (preferred, faster)."""
    from torchcodec.decoders import VideoDecoder

    decoder = VideoDecoder(video_path)
    frames = decoder.get_frames_at(indices=frame_indices).data
    return frames  # (T, C, H, W)


def _get_video_frame_count_torchcodec(video_path: str) -> int:
    """Get total frame count using torchcodec."""
    from torchcodec.decoders import VideoDecoder
    decoder = VideoDecoder(video_path)
    return len(decoder)


def _load_video_decord(video_path: str, frame_indices: np.ndarray) -> torch.Tensor:
    """Load video using decord (fallback)."""
    from decord import VideoReader, cpu

    vr = VideoReader(video_path, ctx=cpu(0), num_threads=1)
    frames = vr.get_batch(frame_indices).asnumpy()
    frames = torch.from_numpy(frames).permute(0, 3, 1, 2)  # (T, H, W, C) -> (T, C, H, W)
    return frames


def _get_video_frame_count_decord(video_path: str) -> int:
    """Get total frame count using decord."""
    from decord import VideoReader, cpu
    vr = VideoReader(video_path, ctx=cpu(0), num_threads=1)
    return len(vr)


def get_video_frame_count(video_path: str) -> int:
    """Get total frame count from video file."""
    try:
        return _get_video_frame_count_torchcodec(video_path)
    except Exception:
        return _get_video_frame_count_decord(video_path)


def load_video(video_path: str, frame_indices: np.ndarray) -> torch.Tensor:
    """
    Load video frames at specified indices.

    Tries torchcodec first (faster), falls back to decord.

    Args:
        video_path: Path to video file
        frame_indices: Array of frame indices to load

    Returns:
        frames: Tensor of shape (T, C, H, W)
    """
    try:
        return _load_video_torchcodec(video_path, frame_indices)
    except Exception:
        try:
            return _load_video_decord(video_path, frame_indices)
        except Exception as e:
            raise RuntimeError(f"Failed to load video {video_path}: {e}")


class SSV2Dataset(IterableDataset):
    """
    Something-Something V2 video classification dataset.

    Loads videos and extracts frames for V-JEPA 2 processing.
    Implements IterableDataset for efficient streaming.
    """

    def __init__(self, config: SSV2DatasetConfig, split: str = "train"):
        super().__init__()
        self.config = config
        self.split = split
        self.is_train = (split == "train")

        # Load annotations
        with open(config.annotations_path, "r") as f:
            annotations = json.load(f)

        # Load label mapping: template -> class_id
        with open(config.labels_path, "r") as f:
            label_json = json.load(f)
            # Labels file format: {"0": "Approaching something with your camera", ...}
            self.label_map = {k: int(v) for k, v in label_json.items()}
            self.id_to_label = {int(v): k for k, v in label_json.items()}

        # Build sample list: (video_id, label)
        self.samples = []
        for ann in annotations:
            video_id = ann["id"]
            template = ann.get("template", ann.get("label"))
            template = template.replace("[", "").replace("]", "")
            if template in self.label_map:
                label = self.label_map[template]
                self.samples.append((video_id, label))
            else:
                raise ValueError(f"could not find label for: {ann}")

        self.metadata = SSV2DatasetMetadata(
            num_classes=174,
            num_frames=config.num_frames,
            frame_size=config.frame_size,
            total_samples=len(self.samples)
        )

        self.local_batch_size = config.global_batch_size // config.num_replicas

        # Random state for reproducibility
        self._rng = np.random.Generator(np.random.Philox(seed=config.seed + config.rank))

        # Set up augmentation pipeline
        if config.augment_config is not None:
            self.augment_config = config.augment_config
        else:
            # Use defaults based on split
            self.augment_config = (
                get_default_train_augment_config() if self.is_train
                else get_default_val_augment_config()
            )
        self.augmentor = VideoAugmentor(self.augment_config, self._rng)

    def _get_video_path(self, video_id: str) -> str:
        """Get full path to video file."""
        # Try common extensions
        for ext in [".webm", ".mp4", ".avi"]:
            path = os.path.join(self.config.data_root, f"{video_id}{ext}")
            if os.path.exists(path):
                return path

        # Try without extension (some datasets use directory structure)
        path = os.path.join(self.config.data_root, video_id)
        if os.path.exists(path):
            return path

        raise FileNotFoundError(f"Video not found: {video_id}")

    def _load_sample(self, video_id: str, label: int) -> Optional[Dict[str, torch.Tensor]]:
        """Load a single video sample with augmentation."""
        try:
            video_path = self._get_video_path(video_id)

            # Get frame count and compute sampling indices
            total_frames = get_video_frame_count(video_path)
            num_frames = self.config.num_frames

            # Apply temporal jittering during training, uniform sampling otherwise
            if self.is_train and self.augment_config.temporal_jitter:
                frame_indices = apply_temporal_jitter(
                    total_frames=total_frames,
                    num_frames=num_frames,
                    rng=self._rng,
                    jitter_range=self.augment_config.temporal_jitter_range
                )
            else:
                frame_indices = uniform_temporal_sample(total_frames, num_frames)

            # Load video frames at computed indices
            frames = load_video(video_path, frame_indices)

            # Apply augmentation pipeline (handles cropping, color jitter, etc.)
            target_size = self.config.frame_size
            frames = self.augmentor(frames, target_size)

            # Ensure proper dtype
            frames = frames.to(torch.uint8)

            return {
                "video_frames": frames,  # (T, C, H, W)
                "labels": torch.tensor(label, dtype=torch.long),
                "video_id": video_id
            }
        except Exception as e:
            print(f"Warning: Failed to load video {video_id}: {e}")
            return None

    def _iter_train(self) -> Iterator[Dict[str, torch.Tensor]]:
        """Training iterator with shuffling."""
        # Get worker info for sharding
        worker_info = torch.utils.data.get_worker_info()

        # Shuffle samples
        indices = self._rng.permutation(len(self.samples))

        # Shard across workers if using multiple workers
        if worker_info is not None:
            indices = indices[worker_info.id::worker_info.num_workers]

        batch_frames = []
        batch_labels = []
        batch_ids = []

        for idx in indices:
            video_id, label = self.samples[idx]
            sample = self._load_sample(video_id, label)

            if sample is not None:
                batch_frames.append(sample["video_frames"])
                batch_labels.append(sample["labels"])
                batch_ids.append(sample["video_id"])

                if len(batch_frames) >= self.local_batch_size:
                    yield {
                        "video_frames": torch.stack(batch_frames),  # (B, T, C, H, W)
                        "labels": torch.stack(batch_labels),  # (B,)
                        "video_ids": batch_ids
                    }
                    batch_frames = []
                    batch_labels = []
                    batch_ids = []

        # Yield remaining (drop last incomplete batch in training)

    def _iter_test(self) -> Iterator[Dict[str, torch.Tensor]]:
        """Test/validation iterator (no shuffling)."""
        # Get worker info for sharding
        worker_info = torch.utils.data.get_worker_info()

        # Shard across workers if using multiple workers
        samples = self.samples
        if worker_info is not None:
            samples = self.samples[worker_info.id::worker_info.num_workers]

        batch_frames = []
        batch_labels = []
        batch_ids = []

        for video_id, label in samples:
            sample = self._load_sample(video_id, label)

            if sample is not None:
                batch_frames.append(sample["video_frames"])
                batch_labels.append(sample["labels"])
                batch_ids.append(sample["video_id"])

                if len(batch_frames) >= self.local_batch_size:
                    yield {
                        "video_frames": torch.stack(batch_frames),
                        "labels": torch.stack(batch_labels),
                        "video_ids": batch_ids
                    }
                    batch_frames = []
                    batch_labels = []
                    batch_ids = []

        # Yield remaining samples
        if batch_frames:
            yield {
                "video_frames": torch.stack(batch_frames),
                "labels": torch.stack(batch_labels),
                "video_ids": batch_ids
            }

    def __iter__(self) -> Iterator[Dict[str, torch.Tensor]]:
        if self.is_train:
            yield from self._iter_train()
        else:
            yield from self._iter_test()


def create_ssv2_dataloader(
    config: SSV2DatasetConfig,
    split: str,
    num_workers: int = 4,
    prefetch_factor: int = 2
) -> Tuple[DataLoader, SSV2DatasetMetadata]:
    """
    Create DataLoader for Something-Something V2.

    Args:
        config: Dataset configuration
        split: "train" or "validation"
        num_workers: Number of data loading workers
        prefetch_factor: Number of batches to prefetch per worker

    Returns:
        dataloader: PyTorch DataLoader
        metadata: Dataset metadata
    """
    dataset = SSV2Dataset(config, split=split)

    dataloader = DataLoader(
        dataset,
        batch_size=None,  # Dataset handles batching
        num_workers=num_workers,
        pin_memory=True,
        prefetch_factor=prefetch_factor if num_workers > 0 else None,
        persistent_workers=num_workers > 0
    )

    return dataloader, dataset.metadata


def create_ssv2_config(
    data_root: str,
    train_annotations: str,
    val_annotations: str,
    labels_path: str,
    global_batch_size: int = 32,
    num_frames: int = 64,
    seed: int = 42,
    rank: int = 0,
    num_replicas: int = 1,
    train_augment_config: Optional[VideoAugmentConfig] = None,
    val_augment_config: Optional[VideoAugmentConfig] = None,
) -> Tuple[SSV2DatasetConfig, SSV2DatasetConfig]:
    """
    Create train and validation dataset configs.

    Args:
        data_root: Path to video files
        train_annotations: Path to training annotations JSON
        val_annotations: Path to validation annotations JSON
        labels_path: Path to labels JSON
        global_batch_size: Total batch size across all GPUs
        num_frames: Number of frames to sample per video
        seed: Random seed
        rank: Current process rank
        num_replicas: Total number of processes
        train_augment_config: Augmentation config for training (None = use defaults)
        val_augment_config: Augmentation config for validation (None = use defaults)

    Returns:
        train_config: Training dataset config
        val_config: Validation dataset config
    """
    train_config = SSV2DatasetConfig(
        seed=seed,
        data_root=data_root,
        annotations_path=train_annotations,
        labels_path=labels_path,
        num_frames=num_frames,
        global_batch_size=global_batch_size,
        rank=rank,
        num_replicas=num_replicas,
        augment_config=train_augment_config,
    )

    val_config = SSV2DatasetConfig(
        seed=seed,
        data_root=data_root,
        annotations_path=val_annotations,
        labels_path=labels_path,
        num_frames=num_frames,
        global_batch_size=global_batch_size,
        rank=rank,
        num_replicas=num_replicas,
        augment_config=val_augment_config,
    )

    return train_config, val_config
