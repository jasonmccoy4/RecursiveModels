from typing import Any, Tuple, Dict, Sequence, Optional

import torch
import torch.nn.functional as F
from torch import nn
import math

IGNORE_LABEL_ID = -100


def s(x, epsilon=1e-30):
    return torch.where(
        x<0,
        1/(1-x+ epsilon),
        x + 1
    )


def log_stablemax(x, dim=-1):
    s_x = s(x)
    return torch.log(s_x/torch.sum(s_x, dim=dim, keepdim=True))


def stablemax_cross_entropy(logits, labels, ignore_index: int = -100, valid_mask=None):
    logprobs = log_stablemax(logits.to(torch.float64), dim=-1)

    if valid_mask is None:
        valid_mask = (labels != ignore_index)
    transformed_labels = torch.where(valid_mask, labels, 0)
    prediction_logprobs = torch.gather(logprobs, index=transformed_labels.to(torch.long).unsqueeze(-1), dim=-1).squeeze(-1)

    return -torch.where(valid_mask, prediction_logprobs, 0)


def softmax_cross_entropy(logits, labels, ignore_index: int = -100):
    # Cast logits to f32
    # Flatten logits
    return F.cross_entropy(logits.to(torch.float32).view(-1, logits.shape[-1]), labels.to(torch.long).view(-1), ignore_index=ignore_index, reduction="none").view(labels.shape)


class ACTLossHead(nn.Module):
    def __init__(self, model: nn.Module, loss_type: str):
        super().__init__()
        self.model = model
        self.loss_fn = globals()[loss_type]
        
    def initial_carry(self, *args, **kwargs):
        return self.model.initial_carry(*args, **kwargs)  # type: ignore

    def forward(
        self,
        return_keys: Sequence[str],
        # Model args
        **model_kwargs,
    ) -> Tuple[Any, torch.Tensor, Dict[str, torch.Tensor], Optional[Dict[str, torch.Tensor]], torch.Tensor]:
        # Model logits
        # B x SeqLen x D
        new_carry, outputs = self.model(**model_kwargs)
        labels = new_carry.current_data["labels"]

        with torch.no_grad():
            # Preds
            outputs["preds"] = torch.argmax(outputs["logits"], dim=-1)

            # Correctness
            mask = (labels != IGNORE_LABEL_ID)
            loss_counts = mask.sum(-1)
            loss_divisor = loss_counts.clamp_min(1).unsqueeze(-1)  # Avoid NaNs in division

            is_correct = mask & (torch.argmax(outputs["logits"], dim=-1) == labels)
            seq_is_correct = is_correct.sum(-1) == loss_counts
            
            # Metrics (halted)
            valid_metrics = new_carry.halted & (loss_counts > 0)
            metrics = {
                "count": valid_metrics.sum(),
                
                "accuracy":       torch.where(valid_metrics, (is_correct.to(torch.float32) / loss_divisor).sum(-1), 0).sum(),
                "exact_accuracy": (valid_metrics & seq_is_correct).sum(),

                "q_halt_accuracy": (valid_metrics & ((outputs["q_halt_logits"] >= 0) == seq_is_correct)).sum(),
                "steps":          torch.where(valid_metrics, new_carry.steps, 0).sum(),
            }

        # Losses

        lm_loss = (self.loss_fn(outputs["logits"], labels, ignore_index=IGNORE_LABEL_ID, valid_mask=mask) / loss_divisor).sum()
        q_halt_loss = F.binary_cross_entropy_with_logits(outputs["q_halt_logits"], seq_is_correct.to(outputs["q_halt_logits"].dtype), reduction="sum")
        metrics.update({
            "lm_loss": lm_loss.detach(),
            "q_halt_loss": q_halt_loss.detach(),
        })
        # Q continue (bootstrapping target loss); Alexia: This fits Q-learning, but seems totally unecessary
        q_continue_loss = 0
        if "target_q_continue" in outputs:
            q_continue_loss = F.binary_cross_entropy_with_logits(outputs["q_continue_logits"], outputs["target_q_continue"], reduction="sum")

            metrics["q_continue_loss"] = q_continue_loss.detach()
        # Filter outputs for return
        detached_outputs = {k: outputs[k].detach() for k in return_keys if k in outputs}

        return new_carry, lm_loss + 0.5 * (q_halt_loss + q_continue_loss), metrics, detached_outputs, new_carry.halted.all()


class VideoClassificationACTLossHead(nn.Module):
    """
    ACT loss head for video classification task.

    Combines classification cross-entropy loss with ACT Q-halt loss.
    The Q-halt loss trains the model to predict whether the classification
    will be correct, enabling adaptive computation.
    """

    def __init__(
        self,
        model: nn.Module,
        label_smoothing: float = 0.1,
        q_loss_weight: float = 0.5
    ):
        super().__init__()
        self.model = model
        self.label_smoothing = label_smoothing
        self.q_loss_weight = q_loss_weight

    def initial_carry(self, *args, **kwargs):
        return self.model.initial_carry(*args, **kwargs)

    def forward(
        self,
        return_keys: Sequence[str],
        carry: Any,
        vjepa_features: torch.Tensor,
        labels: torch.Tensor,
    ) -> Tuple[Any, torch.Tensor, Dict[str, torch.Tensor], Optional[Dict[str, torch.Tensor]], torch.Tensor]:
        """
        Forward pass with loss computation.

        Args:
            return_keys: Keys of outputs to return
            carry: ACT carry state
            vjepa_features: V-JEPA 2 features (B, S, 1024)
            labels: Ground truth class labels (B,)

        Returns:
            new_carry: Updated carry
            loss: Total loss (classification + ACT)
            metrics: Training metrics dict
            outputs: Detached outputs for evaluation
            all_halted: Whether all samples have halted
        """
        # Forward through model
        new_carry, outputs = self.model(
            carry=carry,
            vjepa_features=vjepa_features,
            labels=labels
        )

        logits = outputs["logits"]  # (B, num_classes)
        q_halt_logits = outputs["q_halt_logits"]
        q_continue_logits = outputs["q_continue_logits"]

        with torch.no_grad():
            # Predictions
            preds = torch.argmax(logits, dim=-1)
            outputs["preds"] = preds

            # Correctness for Q-learning target
            is_correct = (preds == new_carry.labels)

            # Metrics (only for halted samples)
            valid_metrics = new_carry.halted
            metrics = {
                "count": valid_metrics.sum(),
                "accuracy": (valid_metrics & is_correct).sum(),
                "q_halt_accuracy": (valid_metrics & ((q_halt_logits >= 0) == is_correct)).sum(),
                "steps": torch.where(valid_metrics, new_carry.steps, 0).sum(),
            }

        # Classification loss (cross-entropy with label smoothing)
        cls_loss = F.cross_entropy(
            logits,
            new_carry.labels,
            label_smoothing=self.label_smoothing,
            reduction="sum"
        )

        # Q-halt loss (binary cross-entropy: predict correctness)
        q_halt_loss = F.binary_cross_entropy_with_logits(
            q_halt_logits,
            is_correct.to(q_halt_logits.dtype),
            reduction="sum"
        )

        # Q-continue loss (optional, same as ACTLossHead)
        q_continue_loss = torch.tensor(0.0, device=logits.device)
        if "target_q_continue" in outputs:
            q_continue_loss = F.binary_cross_entropy_with_logits(
                q_continue_logits,
                outputs["target_q_continue"],
                reduction="sum"
            )
            metrics["q_continue_loss"] = q_continue_loss.detach()

        # Total loss
        total_loss = cls_loss + self.q_loss_weight * (q_halt_loss + q_continue_loss)

        metrics.update({
            "cls_loss": cls_loss.detach(),
            "q_halt_loss": q_halt_loss.detach(),
        })

        # Filter outputs for return
        detached_outputs = {k: outputs[k].detach() for k in return_keys if k in outputs}

        return new_carry, total_loss, metrics, detached_outputs, new_carry.halted.all()

