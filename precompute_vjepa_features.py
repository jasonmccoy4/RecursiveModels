"""
V-JEPA 2 Feature Pre-computation Script

Pre-computes and caches V-JEPA 2 features for all videos in a dataset.
This eliminates the ~70% forward pass overhead of running the frozen V-JEPA 2
encoder during training.

Usage:
    # For Diving48:
    python precompute_vjepa_features.py \
        --dataset diving48 \
        --data_root C:/Users/Jason/ai_data/diving48/videos/rgb \
        --train_annotations C:/Users/Jason/ai_data/diving48/Diving48_train.json \
        --val_annotations C:/Users/Jason/ai_data/diving48/Diving48_test.json \
        --output_dir C:/Users/Jason/ai_data/diving48/vjepa_features

    # For SSv2:
    python precompute_vjepa_features.py \
        --dataset ssv2 \
        --data_root path/to/ssv2/videos \
        --train_annotations path/to/train.json \
        --val_annotations path/to/validation.json \
        --labels_path path/to/labels.json \
        --output_dir path/to/ssv2/vjepa_features
"""

import argparse
import json
import os
from pathlib import Path

import torch
import numpy as np
from tqdm import tqdm

from models.vjepa_extractor import create_vjepa_extractor
from video_dataset import (
    load_video,
    get_video_frame_count,
    uniform_temporal_sample,
)


def get_video_path(data_root: str, video_id: str) -> str:
    """Find video file with various extensions."""
    for ext in [".mp4", ".webm", ".avi", ".mkv"]:
        path = os.path.join(data_root, f"{video_id}{ext}")
        if os.path.exists(path):
            return path
    # Try without extension
    path = os.path.join(data_root, video_id)
    if os.path.exists(path):
        return path
    raise FileNotFoundError(f"Video not found: {video_id}")


def load_annotations(dataset: str, train_path: str, val_path: str, labels_path: str = None):
    """Load video annotations for the dataset."""
    with open(train_path, "r") as f:
        train_annotations = json.load(f)
    with open(val_path, "r") as f:
        val_annotations = json.load(f)

    samples = []

    if dataset == "diving48":
        # Diving48 format: [{"vid_name": "...", "label": int}, ...]
        for ann in train_annotations:
            video_id = ann.get("vid_name", ann.get("video_id", ann.get("id")))
            samples.append({"video_id": video_id, "split": "train"})
        for ann in val_annotations:
            video_id = ann.get("vid_name", ann.get("video_id", ann.get("id")))
            samples.append({"video_id": video_id, "split": "test"})
    else:
        # SSv2 format
        for ann in train_annotations:
            samples.append({"video_id": ann["id"], "split": "train"})
        for ann in val_annotations:
            samples.append({"video_id": ann["id"], "split": "validation"})

    return samples


def precompute_features(
    dataset: str,
    data_root: str,
    train_annotations: str,
    val_annotations: str,
    output_dir: str,
    labels_path: str = None,
    vjepa_model: str = "facebook/vjepa2-vitl-fpc64-256",
    num_frames: int = 64,
    frame_size: int = 256,
    batch_size: int = 4,
):
    """Pre-compute V-JEPA features for all videos."""

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Load V-JEPA extractor
    print(f"Loading V-JEPA 2 model: {vjepa_model}")
    extractor = create_vjepa_extractor(vjepa_model)

    # Load annotations
    print("Loading annotations...")
    samples = load_annotations(dataset, train_annotations, val_annotations, labels_path)
    print(f"Found {len(samples)} videos to process")

    # Track already processed
    processed_count = 0
    skipped_count = 0
    error_count = 0

    # Process videos
    for sample in tqdm(samples, desc="Extracting V-JEPA features"):
        video_id = sample["video_id"]
        output_path = os.path.join(output_dir, f"{video_id}.pt")

        # Skip if already processed
        if os.path.exists(output_path):
            skipped_count += 1
            continue

        try:
            # Get video path
            video_path = get_video_path(data_root, video_id)

            # Sample frames uniformly
            total_frames = get_video_frame_count(video_path)
            frame_indices = uniform_temporal_sample(total_frames, num_frames)

            # Load frames
            frames = load_video(video_path, frame_indices)  # (T, C, H, W)

            # Preprocess for V-JEPA
            processed = extractor.preprocess(frames, return_tensors="pt")
            pixel_values = processed["pixel_values_videos"].to("cuda")

            # Extract features
            with torch.no_grad():
                features = extractor(pixel_values)  # (1, seq_len, 1024)
                features = features.squeeze(0).cpu().to(torch.bfloat16)  # (seq_len, 1024)

            # Save features
            torch.save(features, output_path)
            processed_count += 1

        except Exception as e:
            print(f"\nError processing {video_id}: {e}")
            error_count += 1
            continue

    print(f"\nDone!")
    print(f"  Processed: {processed_count}")
    print(f"  Skipped (already exists): {skipped_count}")
    print(f"  Errors: {error_count}")

    # Save metadata
    metadata = {
        "dataset": dataset,
        "vjepa_model": vjepa_model,
        "num_frames": num_frames,
        "frame_size": frame_size,
        "feature_dim": 1024,
        "dtype": "bfloat16",
        "total_videos": len(samples),
        "processed": processed_count + skipped_count,
        "errors": error_count,
    }
    with open(os.path.join(output_dir, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"Metadata saved to {output_dir}/metadata.json")


def main():
    parser = argparse.ArgumentParser(description="Pre-compute V-JEPA 2 features")
    parser.add_argument("--dataset", type=str, required=True, choices=["diving48", "ssv2"])
    parser.add_argument("--data_root", type=str, required=True, help="Path to video files")
    parser.add_argument("--train_annotations", type=str, required=True)
    parser.add_argument("--val_annotations", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory for features")
    parser.add_argument("--labels_path", type=str, default=None, help="Path to labels (for SSv2)")
    parser.add_argument("--vjepa_model", type=str, default="facebook/vjepa2-vitl-fpc64-256")
    parser.add_argument("--num_frames", type=int, default=64)
    parser.add_argument("--frame_size", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=4)

    args = parser.parse_args()

    precompute_features(
        dataset=args.dataset,
        data_root=args.data_root,
        train_annotations=args.train_annotations,
        val_annotations=args.val_annotations,
        output_dir=args.output_dir,
        labels_path=args.labels_path,
        vjepa_model=args.vjepa_model,
        num_frames=args.num_frames,
        frame_size=args.frame_size,
        batch_size=args.batch_size,
    )


if __name__ == "__main__":
    main()
