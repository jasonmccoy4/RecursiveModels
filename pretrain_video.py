"""
V-JEPA 2 Recursive Video Classifier Training Script

Trains the recursive video classifier on video classification datasets
using V-JEPA 2 as a frozen feature extractor.

Supported datasets:
- Something-Something V2 (ssv2): 174 classes, direction-sensitive actions
- Diving48: 48 fine-grained diving action classes

Usage:
    # SSv2 (default):
    python pretrain_video.py arch=trm_vjepa

    # Diving48:
    python pretrain_video.py --config-name=cfg_video_pretrain_diving48 arch=trm_vjepa_diving48

    # With custom data paths:
    python pretrain_video.py arch=trm_vjepa \
        dataset=ssv2 \
        data_root=path/to/ssv2/videos \
        train_annotations=path/to/train.json \
        val_annotations=path/to/validation.json \
        labels_path=path/to/labels.json
"""

from typing import Optional, Any, List
from dataclasses import dataclass
import os
import sys
import math

import torch
import torch.distributed as dist
from torch import nn
from torch.utils.data import DataLoader

import tqdm
import wandb
import coolname
import hydra
import pydantic
from omegaconf import DictConfig
from adam_atan2_pytorch import AdamAtan2

from video_dataset import (
    SSV2DatasetConfig, SSV2DatasetMetadata,
    create_ssv2_dataloader, create_ssv2_config,
    Diving48DatasetConfig, Diving48DatasetMetadata,
    create_diving48_dataloader, create_diving48_config,
)
from models.vjepa_extractor import VJEPA2FeatureExtractor, create_vjepa_extractor
from utils.functions import load_model_class

torch.set_float32_matmul_precision('high')

class LossConfig(pydantic.BaseModel):
    model_config = pydantic.ConfigDict(extra='allow')
    name: str


class ArchConfig(pydantic.BaseModel):
    model_config = pydantic.ConfigDict(extra='allow')
    name: str
    loss: LossConfig


class VideoPretrainConfig(pydantic.BaseModel):
    """Configuration for video classification training."""

    # Model architecture
    arch: ArchConfig

    # Dataset selection: "ssv2" or "diving48"
    dataset: str = "ssv2"

    # Data size to optimise parameters on less data
    data_size: float = 1.0

    # Data paths
    data_root: str = "data/ssv2/videos"
    train_annotations: str = "data/ssv2/something-something-v2-train.json"
    val_annotations: str = "data/ssv2/something-something-v2-validation.json"
    labels_path: str = "data/ssv2/something-something-v2-labels.json"

    # V-JEPA 2 config
    vjepa_model_name: str = "facebook/vjepa2-vitl-fpc64-256"

    # Video config
    num_frames: int = 64
    frame_size: int = 256

    # Training hyperparams
    global_batch_size: int = 32
    epochs: int = 50

    lr: float = 1e-4
    lr_min_ratio: float = 0.1
    lr_warmup_steps: int = 1000

    weight_decay: float = 0.05
    beta1: float = 0.9
    beta2: float = 0.999

    # Mixed precision
    mixed_precision: bool = True

    # Gradient accumulation
    gradient_accumulation_steps: int = 1

    # DataLoader
    num_workers: int = 4
    prefetch_factor: int = 2
    persistent_workers: bool = True

    # Pre-computed V-JEPA features (for 3-5x speedup)
    precomputed_features_dir: Optional[str] = None

    # Names and checkpointing
    project_name: Optional[str] = None
    run_name: Optional[str] = None
    checkpoint_path: Optional[str] = None
    load_checkpoint: Optional[str] = None

    # Evaluation
    eval_interval: int = 5  # Evaluate every N epochs
    checkpoint_every_eval: bool = True

    # Extras
    seed: int = 42


@dataclass
class TrainState:
    model: nn.Module
    vjepa_extractor: VJEPA2FeatureExtractor
    optimizer: torch.optim.Optimizer
    carry: Any

    step: int
    epoch: int
    total_steps: int


def cosine_schedule_with_warmup(
    current_step: int,
    base_lr: float,
    num_warmup_steps: int,
    num_training_steps: int,
    min_ratio: float = 0.1
) -> float:
    """Compute learning rate with warmup and cosine decay."""
    if current_step < num_warmup_steps:
        return base_lr * float(current_step) / float(max(1, num_warmup_steps))

    progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
    return base_lr * (min_ratio + max(0.0, (1 - min_ratio) * 0.5 * (1.0 + math.cos(math.pi * progress))))


def create_video_dataloader(
    config: VideoPretrainConfig,
    split: str,
    rank: int,
    world_size: int,
    num_workers: int = 4,
    prefetch_factor: int = 2,
    persistent_workers: bool = True,
) -> tuple:
    """Create video dataloader for the configured dataset.

    Supports:
    - SSv2: Uses optimized augmentations (no horizontal flip due to direction-sensitive labels)
    - Diving48: Fine-grained diving actions (allows horizontal flip)

    Training augmentations include: temporal jittering, random resized crop, color jitter,
    random grayscale, gaussian blur, and random erasing.

    If precomputed_features_dir is set, loads pre-computed V-JEPA features instead of raw videos.
    """
    # Check for precomputed features
    if config.precomputed_features_dir:
        from video_dataset import (
            create_precomputed_diving48_dataloader,
            create_precomputed_ssv2_dataloader,
            PrecomputedDatasetConfig,
        )
        precomputed_config = PrecomputedDatasetConfig(
            seed=config.seed,
            features_dir=config.precomputed_features_dir,
            annotations_path=config.train_annotations if split == "train" else config.val_annotations,
            global_batch_size=config.global_batch_size,
            rank=rank,
            num_replicas=world_size,
            data_size=config.data_size,
        )
        if config.dataset == "diving48":
            diving48_split = "test" if split == "validation" else split
            return create_precomputed_diving48_dataloader(
                precomputed_config, diving48_split,
                num_workers=num_workers, prefetch_factor=prefetch_factor,
                persistent_workers=persistent_workers
            )
        else:
            return create_precomputed_ssv2_dataloader(
                precomputed_config, split,
                num_workers=num_workers, prefetch_factor=prefetch_factor,
                persistent_workers=persistent_workers
            )

    if config.dataset == "diving48":
        # Map "validation" split to "test" for Diving48
        diving48_split = "test" if split == "validation" else split
        dataset_config = Diving48DatasetConfig(
            seed=config.seed,
            data_root=config.data_root,
            annotations_path=config.train_annotations if split == "train" else config.val_annotations,
            labels_path=config.labels_path if config.labels_path else None,
            num_frames=config.num_frames,
            frame_size=config.frame_size,
            global_batch_size=config.global_batch_size,
            rank=rank,
            num_replicas=world_size,
            data_size=config.data_size,
        )
        return create_diving48_dataloader(
            dataset_config, diving48_split,
            num_workers=num_workers, prefetch_factor=prefetch_factor,
            persistent_workers=persistent_workers
        )
    else:
        # Default: SSv2
        dataset_config = SSV2DatasetConfig(
            seed=config.seed,
            data_root=config.data_root,
            annotations_path=config.train_annotations if split == "train" else config.val_annotations,
            labels_path=config.labels_path,
            num_frames=config.num_frames,
            frame_size=config.frame_size,
            global_batch_size=config.global_batch_size,
            rank=rank,
            num_replicas=world_size,
            data_size=config.data_size,
        )
        return create_ssv2_dataloader(
            dataset_config, split,
            num_workers=num_workers, prefetch_factor=prefetch_factor,
            persistent_workers=persistent_workers
        )


def create_model(
    config: VideoPretrainConfig,
    metadata: SSV2DatasetMetadata,
    rank: int,
    world_size: int
) -> tuple:
    """Create recursive classifier model with loss head."""
    model_cfg = dict(
        **config.arch.__pydantic_extra__,
        batch_size=config.global_batch_size // world_size,
    )

    # Instantiate model with loss head
    model_cls = load_model_class(config.arch.name)
    loss_head_cls = load_model_class(config.arch.loss.name)

    with torch.device("cuda"):
        model = model_cls(model_cfg)
        print(model)
        model = loss_head_cls(model, **config.arch.loss.__pydantic_extra__)

        # Compile for performance (optional)
        # Note: "max-autotune" has overflow bugs on Windows, use "reduce-overhead" instead
        if "DISABLE_COMPILE" not in os.environ:
            model = torch.compile(model)

        # Load checkpoint if specified
        if config.load_checkpoint and rank == 0:
            print(f"Loading checkpoint: {config.load_checkpoint}")
            state_dict = torch.load(config.load_checkpoint, map_location="cuda")
            model.load_state_dict(state_dict, strict=False)

        # Broadcast parameters from rank 0
        if world_size > 1:
            with torch.no_grad():
                for param in list(model.parameters()) + list(model.buffers()):
                    dist.broadcast(param, src=0)

    # Create optimizer
    optimizer = AdamAtan2(
        model.parameters(),
        lr=config.lr,
        weight_decay=config.weight_decay,
        betas=(config.beta1, config.beta2)
    )

    return model, optimizer


def train_batch(
    config: VideoPretrainConfig,
    train_state: TrainState,
    batch: dict,
    world_size: int
) -> Optional[dict]:
    """Process a single training batch."""
    train_state.step += 1

    labels = batch["labels"].cuda(non_blocking=True)  # (B,)

    # Check if using pre-computed features or raw video
    if "vjepa_features" in batch:
        # Pre-computed features path (3-5x faster)
        vjepa_features = batch["vjepa_features"].cuda(non_blocking=True)
        vjepa_features = vjepa_features.to(torch.bfloat16)
        batch_size = vjepa_features.shape[0]
    else:
        # Raw video path - extract V-JEPA features
        video_frames = batch["video_frames"].cuda(non_blocking=True)  # (B, T, C, H, W)
        batch_size = video_frames.shape[0]

        # Preprocess for V-JEPA 2
        processed = train_state.vjepa_extractor.preprocess(
            video_frames,
            return_tensors="pt"
        )
        pixel_values = processed["pixel_values_videos"].to("cuda", non_blocking=True)

        # Extract V-JEPA 2 features (frozen, no gradients)
        with torch.no_grad():
            vjepa_features = train_state.vjepa_extractor(pixel_values)  # (B, S, 1024)
            vjepa_features = vjepa_features.to(torch.bfloat16)

    seq_len = vjepa_features.shape[1]

    # Initialize carry if needed
    if train_state.carry is None:
        with torch.device("cuda"):
            train_state.carry = train_state.model.initial_carry(
                batch_size=batch_size,
                seq_len=seq_len,
                vjepa_features=vjepa_features,
                labels=labels
            )

    # Forward through model with loss
    train_state.carry, loss, metrics, _, all_halted = train_state.model(
        carry=train_state.carry,
        vjepa_features=vjepa_features,
        labels=labels,
        return_keys=[]
    )

    # Backward
    scaled_loss = loss / (config.global_batch_size * config.gradient_accumulation_steps)
    scaled_loss.backward()

    # Gradient accumulation check
    if train_state.step % config.gradient_accumulation_steps == 0:
        # Allreduce gradients
        if world_size > 1:
            for param in train_state.model.parameters():
                if param.grad is not None:
                    dist.all_reduce(param.grad)

        # Update learning rate
        lr = cosine_schedule_with_warmup(
            current_step=train_state.step // config.gradient_accumulation_steps,
            base_lr=config.lr,
            num_warmup_steps=config.lr_warmup_steps,
            num_training_steps=train_state.total_steps,
            min_ratio=config.lr_min_ratio
        )
        for param_group in train_state.optimizer.param_groups:
            param_group['lr'] = lr

        # Optimizer step
        train_state.optimizer.step()
        train_state.optimizer.zero_grad()

    # Reduce and return metrics
    if len(metrics):
        metric_keys = list(sorted(metrics.keys()))
        metric_values = torch.stack([metrics[k].float() for k in metric_keys])

        if world_size > 1:
            dist.reduce(metric_values, dst=0)

        metric_values = metric_values.cpu().numpy()
        count = max(metric_values[metric_keys.index("count")], 1)

        reduced_metrics = {}
        for i, k in enumerate(metric_keys):
            if k.endswith("loss"):
                reduced_metrics[f"train/{k}"] = metric_values[i] / config.global_batch_size
            elif k != "count":
                reduced_metrics[f"train/{k}"] = metric_values[i] / count

        reduced_metrics["train/lr"] = lr if train_state.step % config.gradient_accumulation_steps == 0 else None

        return reduced_metrics

    return None


def evaluate(
    config: VideoPretrainConfig,
    train_state: TrainState,
    eval_loader: DataLoader,
    rank: int,
    world_size: int
) -> Optional[dict]:
    """Run evaluation on validation set."""
    train_state.model.eval()

    total_correct = 0
    total_samples = 0
    total_loss = 0.0
    total_steps = 0

    with torch.inference_mode():
        for batch in eval_loader:
            labels = batch["labels"].cuda(non_blocking=True)
            batch_size = labels.shape[0]

            # Check if using pre-computed features or raw video
            if "vjepa_features" in batch:
                # Pre-computed features path
                vjepa_features = batch["vjepa_features"].cuda(non_blocking=True)
                vjepa_features = vjepa_features.to(torch.bfloat16)
            else:
                # Raw video path - extract V-JEPA features
                video_frames = batch["video_frames"].cuda(non_blocking=True)

                processed = train_state.vjepa_extractor.preprocess(
                    video_frames,
                    return_tensors="pt"
                )
                pixel_values = processed["pixel_values_videos"].to("cuda", non_blocking=True)
                vjepa_features = train_state.vjepa_extractor(pixel_values)
                vjepa_features = vjepa_features.to(torch.bfloat16)

            seq_len = vjepa_features.shape[1]

            # Initialize carry
            carry = train_state.model.initial_carry(
                batch_size=batch_size,
                seq_len=seq_len,
                vjepa_features=vjepa_features,
                labels=labels
            )

            # Run ACT until all halted
            steps = 0
            while True:
                carry, loss, metrics, outputs, all_halted = train_state.model(
                    carry=carry,
                    vjepa_features=vjepa_features,
                    labels=labels,
                    return_keys=["preds"]
                )
                steps += 1

                if all_halted or steps >= config.arch.halt_max_steps:
                    break

            # Accumulate metrics
            preds = outputs.get("preds", torch.argmax(metrics.get("logits", torch.zeros(1)), dim=-1))
            correct = (preds == labels).sum().item()

            total_correct += correct
            total_samples += batch_size
            total_loss += loss.item()
            total_steps += steps

    train_state.model.train()

    # Aggregate across ranks
    if world_size > 1:
        # Barrier to ensure all ranks have finished processing their batches
        # (different ranks may have different batch counts due to uneven data sharding)
        dist.barrier()
        stats = torch.tensor([total_correct, total_samples, total_loss, total_steps], device="cuda")
        dist.reduce(stats, dst=0)
        total_correct, total_samples, total_loss, total_steps = stats.tolist()

    if rank == 0 and total_samples > 0:
        return {
            "val/accuracy": total_correct / total_samples,
            "val/loss": total_loss / total_samples,
            "val/avg_steps": total_steps / (total_samples / config.global_batch_size)
        }

    return None


def save_checkpoint(config: VideoPretrainConfig, train_state: TrainState):
    """Save model checkpoint."""
    if config.checkpoint_path is None:
        return

    os.makedirs(config.checkpoint_path, exist_ok=True)
    path = os.path.join(config.checkpoint_path, f"epoch_{train_state.epoch}.pt")
    torch.save(train_state.model.state_dict(), path)
    print(f"Saved checkpoint: {path}")


def load_synced_config(hydra_config: DictConfig, rank: int, world_size: int) -> VideoPretrainConfig:
    """Load and sync config across processes."""
    objects = [None]
    if rank == 0:
        config = VideoPretrainConfig(**hydra_config)

        # Auto-generate names based on dataset
        dataset_name = config.dataset.upper()
        if config.project_name is None:
            config.project_name = f"{dataset_name}-VJEPA-RecursiveClassifier"
        if config.run_name is None:
            config.run_name = f"vjepa-{config.dataset}-{coolname.generate_slug(2)}"
        if config.checkpoint_path is None:
            config.checkpoint_path = os.path.join("checkpoints", config.project_name, config.run_name)

        objects = [config]

    if world_size > 1:
        dist.broadcast_object_list(objects, src=0)

    return objects[0]


@hydra.main(config_path="config", config_name="cfg_video_pretrain_ssv2_conv", version_base=None)
def launch(hydra_config: DictConfig):
    """Main training entry point."""
    RANK = 0
    WORLD_SIZE = 1

    # Initialize distributed training
    if "LOCAL_RANK" in os.environ:
        # Use gloo on Windows (NCCL not supported), nccl on Linux
        backend = "gloo" if sys.platform == "win32" else "nccl"
        dist.init_process_group(backend=backend)
        RANK = dist.get_rank()
        WORLD_SIZE = dist.get_world_size()
        torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))

    # Load config
    config = load_synced_config(hydra_config, rank=RANK, world_size=WORLD_SIZE)

    # Seed
    torch.random.manual_seed(config.seed + RANK)

    # Create dataloaders
    train_loader, train_metadata = create_video_dataloader(
        config, "train", RANK, WORLD_SIZE, config.num_workers,
        prefetch_factor=config.prefetch_factor, persistent_workers=config.persistent_workers
    )
    val_loader, val_metadata = create_video_dataloader(
        config, "validation", RANK, WORLD_SIZE, config.num_workers,
        prefetch_factor=config.prefetch_factor, persistent_workers=config.persistent_workers
    )

    # Create V-JEPA 2 feature extractor (frozen)
    if RANK == 0:
        print(f"Loading V-JEPA 2: {config.vjepa_model_name}")
    vjepa_extractor = create_vjepa_extractor(
        model_name=config.vjepa_model_name,
        dtype=torch.float16 if config.mixed_precision else torch.float32
    )

    # Create model
    model, optimizer = create_model(config, train_metadata, RANK, WORLD_SIZE)

    # Estimate total steps and batches
    estimated_samples_per_epoch = train_metadata.total_samples
    batches_per_epoch = estimated_samples_per_epoch // config.global_batch_size
    total_batches = batches_per_epoch * config.epochs
    total_steps = total_batches // config.gradient_accumulation_steps

    if RANK == 0:
        print(f"Estimated batches per epoch: {batches_per_epoch}")
        print(f"Total batches: {total_batches}")
        print(f"Total optimizer steps: {total_steps}")

    # Training state
    train_state = TrainState(
        model=model,
        vjepa_extractor=vjepa_extractor,
        optimizer=optimizer,
        carry=None,
        step=0,
        epoch=0,
        total_steps=total_steps
    )

    # Logger
    if RANK == 0:
        wandb.init(
            project=config.project_name,
            name=config.run_name,
            config=config.model_dump()
        )
        wandb.log({"num_params": sum(p.numel() for p in model.parameters())}, step=0)

    # Training loop
    progress_bar = None
    if RANK == 0:
        progress_bar = tqdm.tqdm(total=batches_per_epoch, desc="Training")

    for epoch in range(config.epochs):
        train_state.epoch = epoch

        # Set epoch for proper shuffling in distributed training
        train_loader.dataset.set_epoch(epoch)

        if RANK == 0:
            print(f"\n=== Epoch {epoch + 1}/{config.epochs} ===")

        # Train
        train_state.model.train()
        for batch in train_loader:
            metrics = train_batch(config, train_state, batch, WORLD_SIZE)

            if RANK == 0 and metrics is not None:
                wandb.log(metrics, step=train_state.step)
                if progress_bar is not None:
                    progress_bar.update(1)

        # Evaluate
        if (epoch + 1) % config.eval_interval == 0:
            # Barrier to ensure all ranks finish training before eval
            if WORLD_SIZE > 1:
                dist.barrier()
            if RANK == 0:
                print("Running evaluation...")
            metrics = evaluate(config, train_state, val_loader, RANK, WORLD_SIZE)

            if RANK == 0 and metrics is not None:
                wandb.log(metrics, step=train_state.step)
                print(f"Validation accuracy: {metrics['val/accuracy']:.4f}")

            # Checkpoint
            if RANK == 0 and config.checkpoint_every_eval:
                save_checkpoint(config, train_state)

    # Final checkpoint
    if RANK == 0:
        save_checkpoint(config, train_state)
        print("Training complete!")

    # Cleanup
    if dist.is_initialized():
        dist.destroy_process_group()
    if RANK == 0:
        wandb.finish()


if __name__ == "__main__":
    launch()
