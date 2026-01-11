"""
V-JEPA 2 Feature Extractor Wrapper

Provides a frozen V-JEPA 2 encoder for extracting video features.
The model is kept frozen (no gradient updates) and used as a feature extractor
for downstream recursive classification.
"""

from typing import Optional
import torch
from torch import nn


class VJEPA2FeatureExtractor(nn.Module):
    """
    Frozen V-JEPA 2 feature extractor wrapper.

    Extracts spatial-temporal features from video using the V-JEPA 2 encoder.
    All parameters are frozen - no gradients flow through this module.

    Args:
        model_name: HuggingFace model identifier for V-JEPA 2
        device: Device to load the model on
        dtype: Data type for model weights (float16 recommended for efficiency)
    """

    def __init__(
        self,
        model_name: str = "facebook/vjepa2-vitl-fpc64-256",
        device: str = "cuda",
        dtype: torch.dtype = torch.float16
    ):
        super().__init__()

        from transformers import AutoModel, AutoVideoProcessor

        self.model_name = model_name
        self.device = device
        self.dtype = dtype

        # Load video processor for preprocessing
        self.processor = AutoVideoProcessor.from_pretrained(model_name)

        # Load V-JEPA 2 model
        self.model = AutoModel.from_pretrained(
            model_name,
            torch_dtype=dtype,
            attn_implementation="sdpa"  # Use scaled dot-product attention
        ).to(device)

        # Freeze all parameters
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad = False

        # Store config for reference
        self.hidden_size = self.model.config.hidden_size  # 1024 for ViT-L
        self.num_frames = self.model.config.frames_per_clip  # 64

    @torch.no_grad()
    def forward(self, pixel_values_videos: torch.Tensor) -> torch.Tensor:
        """
        Extract features from preprocessed video frames.

        Args:
            pixel_values_videos: Preprocessed video tensor from processor
                Shape: (B, T, C, H, W) where T=num_frames (typically 64)

        Returns:
            features: V-JEPA 2 encoder hidden states
                Shape: (B, seq_len, hidden_size) where hidden_size=1024 for ViT-L
                seq_len depends on spatial/temporal patching configuration
        """
        outputs = self.model(
            pixel_values_videos=pixel_values_videos,
            skip_predictor=True  # Only need encoder features, not predictor
        )
        return outputs.last_hidden_state

    def preprocess(
        self,
        video_frames: torch.Tensor,
        return_tensors: str = "pt"
    ) -> dict:
        """
        Preprocess raw video frames using V-JEPA 2 processor.

        Args:
            video_frames: Raw video frames
                Shape: (B, T, C, H, W) batched or (T, H, W, C) / (T, C, H, W) single
            return_tensors: Return type ("pt" for PyTorch tensors)

        Returns:
            Preprocessed inputs dict with 'pixel_values_videos' key
                Shape will be squeezed to (B, T, C, H, W)
        """
        result = self.processor(video_frames, return_tensors=return_tensors)

        # Squeeze extra dimensions added by processor for batched input
        # Processor may output [1, 1, B, T, C, H, W], we need [B, T, C, H, W]
        pixel_values = result["pixel_values_videos"]
        while pixel_values.dim() > 5:
            pixel_values = pixel_values.squeeze(0)
        result["pixel_values_videos"] = pixel_values

        return result

    def extract_features(self, video_frames: torch.Tensor) -> torch.Tensor:
        """
        Convenience method: preprocess and extract features in one call.

        Args:
            video_frames: Raw video frames (T, H, W, C) or (T, C, H, W)

        Returns:
            features: (1, seq_len, hidden_size) encoder features
        """
        processed = self.preprocess(video_frames)
        pixel_values = processed["pixel_values_videos"].to(self.device)
        return self.forward(pixel_values)

    def get_output_dim(self) -> int:
        """Return the output feature dimension."""
        return self.hidden_size

    def get_expected_frames(self) -> int:
        """Return the expected number of input frames."""
        return self.num_frames


def create_vjepa_extractor(
    model_name: str = "facebook/vjepa2-vitl-fpc64-256",
    device: Optional[str] = None,
    dtype: Optional[torch.dtype] = None
) -> VJEPA2FeatureExtractor:
    """
    Factory function to create a V-JEPA 2 feature extractor.

    Args:
        model_name: HuggingFace model identifier
        device: Device (defaults to "cuda" if available)
        dtype: Data type (defaults to float16)

    Returns:
        Initialized VJEPA2FeatureExtractor
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if dtype is None:
        dtype = torch.float16 if device == "cuda" else torch.float32

    return VJEPA2FeatureExtractor(
        model_name=model_name,
        device=device,
        dtype=dtype
    )
