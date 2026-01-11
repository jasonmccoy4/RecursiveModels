"""
V-JEPA 2 Recursive Video Classifier

Adapts the TinyRecursiveReasoningModel (TRM) architecture for video classification
using V-JEPA 2 as a frozen feature extractor. Key changes from original TRM:

1. Input: V-JEPA 2 features (1024-dim) instead of token embeddings
2. Hidden size: 1024 (matched to V-JEPA 2)
3. Output: Classification head (174 classes) instead of LM head
4. Pooling: Cross-attention over V-JEPA 2 features using evolved query vector
5. ACT: Retained for adaptive computation
"""

from typing import Tuple, Dict
from dataclasses import dataclass
import math
import torch
import torch.nn.functional as F
from torch import nn
from pydantic import BaseModel

from models.common import trunc_normal_init_
from models.layers import (
    rms_norm, SwiGLU, Attention, RotaryEmbedding, CosSin,
    CastedLinear, CrossAttentionPooler
)


@dataclass
class VJEPARecursiveCarry_Inner:
    """Inner carry state for recursive reasoning."""
    z_H: torch.Tensor  # High-level reasoning state: (B, S, D)
    z_L: torch.Tensor  # Low-level reasoning state: (B, S, D)
    query: torch.Tensor  # Evolved query vector: (B, 1, D)


@dataclass
class VJEPARecursiveCarry:
    """Full carry state including ACT control."""
    inner_carry: VJEPARecursiveCarry_Inner

    steps: torch.Tensor  # Step counter per sample: (B,)
    halted: torch.Tensor  # Halt flag per sample: (B,)

    vjepa_features: torch.Tensor  # Cached V-JEPA features: (B, S, D)
    labels: torch.Tensor  # Ground truth labels: (B,)


class VJEPARecursiveConfig(BaseModel):
    """Configuration for V-JEPA 2 Recursive Classifier."""

    batch_size: int

    # V-JEPA 2 config
    vjepa_hidden_size: int = 1024  # V-JEPA 2 ViT-L output dimension

    # Recursive model config
    hidden_size: int = 1024  # Match V-JEPA 2
    H_cycles: int = 3
    L_cycles: int = 6
    L_layers: int = 2

    # Transformer config
    expansion: float = 4.0
    num_heads: int = 16  # 1024 / 64 = 16 heads
    pos_encodings: str = "rope"
    rms_norm_eps: float = 1e-5
    rope_theta: float = 10000.0

    # ACT config
    halt_max_steps: int = 8
    halt_exploration_prob: float = 0.1
    no_ACT_continue: bool = True

    # Classification config
    num_classes: int = 174  # Something-Something V2

    # Cross-attention config
    cross_attn_heads: int = 8

    forward_dtype: str = "bfloat16"


class VJEPARecursiveBlock(nn.Module):
    """Single transformer block for recursive reasoning."""

    def __init__(self, config: VJEPARecursiveConfig) -> None:
        super().__init__()
        self.config = config

        self.self_attn = Attention(
            hidden_size=config.hidden_size,
            head_dim=config.hidden_size // config.num_heads,
            num_heads=config.num_heads,
            num_key_value_heads=config.num_heads,
            causal=False
        )
        self.mlp = SwiGLU(
            hidden_size=config.hidden_size,
            expansion=config.expansion,
        )
        self.norm_eps = config.rms_norm_eps

    def forward(self, cos_sin: CosSin, hidden_states: torch.Tensor) -> torch.Tensor:
        # Post-norm architecture (same as TRM)
        # Self Attention
        hidden_states = rms_norm(
            hidden_states + self.self_attn(cos_sin=cos_sin, hidden_states=hidden_states),
            variance_epsilon=self.norm_eps
        )
        # Feed-forward
        out = self.mlp(hidden_states)
        hidden_states = rms_norm(hidden_states + out, variance_epsilon=self.norm_eps)
        return hidden_states


class VJEPARecursiveReasoningModule(nn.Module):
    """Reasoning module containing multiple transformer blocks."""

    def __init__(self, layers: list):
        super().__init__()
        self.layers = nn.ModuleList(layers)

    def forward(
        self,
        hidden_states: torch.Tensor,
        input_injection: torch.Tensor,
        **kwargs
    ) -> torch.Tensor:
        hidden_states = hidden_states + input_injection
        for layer in self.layers:
            hidden_states = layer(hidden_states=hidden_states, **kwargs)
        return hidden_states


class VJEPARecursiveClassifier_ACTV1_Inner(nn.Module):
    """
    Inner model for V-JEPA 2 recursive video classification.

    Takes V-JEPA 2 features and performs recursive reasoning to produce
    classification logits and ACT halting decisions.
    """

    def __init__(self, config: VJEPARecursiveConfig) -> None:
        super().__init__()
        self.config = config
        self.forward_dtype = getattr(torch, self.config.forward_dtype)

        # Input projection: V-JEPA features -> hidden_size
        # (V-JEPA is 1024, hidden is 1024, so this is identity-like but allows different dims)
        self.input_proj = CastedLinear(
            config.vjepa_hidden_size,
            config.hidden_size,
            bias=False
        )

        # Learnable query token for cross-attention (replaces puzzle_emb)
        self.query_token = nn.Parameter(
            trunc_normal_init_(
                torch.empty(1, 1, config.hidden_size, dtype=self.forward_dtype),
                std=1.0 / math.sqrt(config.hidden_size)
            )
        )

        # Query update projection (updates query based on z_H)
        self.query_update = CastedLinear(config.hidden_size, config.hidden_size, bias=False)

        # Position encodings for video features
        if config.pos_encodings == "rope":
            self.rotary_emb = RotaryEmbedding(
                dim=config.hidden_size // config.num_heads,
                max_position_embeddings=8192,  # Large enough for V-JEPA sequences
                base=config.rope_theta
            )

        # Recursive reasoning layers (L-level, same structure as TRM)
        self.L_level = VJEPARecursiveReasoningModule(
            layers=[VJEPARecursiveBlock(config) for _ in range(config.L_layers)]
        )

        # Initial states for H and L
        self.H_init = nn.Buffer(
            trunc_normal_init_(
                torch.empty(config.hidden_size, dtype=self.forward_dtype),
                std=1
            ),
            persistent=True
        )
        self.L_init = nn.Buffer(
            trunc_normal_init_(
                torch.empty(config.hidden_size, dtype=self.forward_dtype),
                std=1
            ),
            persistent=True
        )

        # Cross-attention pooling module
        self.cross_attention = CrossAttentionPooler(
            query_dim=config.hidden_size,
            key_value_dim=config.vjepa_hidden_size,
            num_heads=config.cross_attn_heads,
            hidden_size=config.hidden_size
        )

        # Classification head (replaces lm_head)
        self.classifier = CastedLinear(config.hidden_size, config.num_classes, bias=True)

        # ACT Q-head (same as TRM)
        self.q_head = CastedLinear(config.hidden_size, 2, bias=True)

        # Q-head initialization (same as TRM)
        with torch.no_grad():
            self.q_head.weight.zero_()
            self.q_head.bias.fill_(-5)

    def _project_input(self, vjepa_features: torch.Tensor) -> torch.Tensor:
        """Project V-JEPA 2 features to hidden dimension."""
        return self.input_proj(vjepa_features)

    def empty_carry(self, batch_size: int, seq_len: int) -> VJEPARecursiveCarry_Inner:
        """Create empty inner carry state."""
        return VJEPARecursiveCarry_Inner(
            z_H=torch.empty(
                batch_size, seq_len, self.config.hidden_size,
                dtype=self.forward_dtype
            ),
            z_L=torch.empty(
                batch_size, seq_len, self.config.hidden_size,
                dtype=self.forward_dtype
            ),
            query=torch.empty(
                batch_size, 1, self.config.hidden_size,
                dtype=self.forward_dtype
            ),
        )

    def reset_carry(
        self,
        reset_flag: torch.Tensor,
        carry: VJEPARecursiveCarry_Inner,
        seq_len: int
    ) -> VJEPARecursiveCarry_Inner:
        """Reset carry state for halted sequences."""
        batch_size = reset_flag.shape[0]

        # Expand init states to sequence length
        H_init_expanded = self.H_init.unsqueeze(0).unsqueeze(0).expand(batch_size, seq_len, -1)
        L_init_expanded = self.L_init.unsqueeze(0).unsqueeze(0).expand(batch_size, seq_len, -1)
        query_init = self.query_token.expand(batch_size, -1, -1)

        return VJEPARecursiveCarry_Inner(
            z_H=torch.where(reset_flag.view(-1, 1, 1), H_init_expanded, carry.z_H),
            z_L=torch.where(reset_flag.view(-1, 1, 1), L_init_expanded, carry.z_L),
            query=torch.where(reset_flag.view(-1, 1, 1), query_init, carry.query),
        )

    def forward(
        self,
        carry: VJEPARecursiveCarry_Inner,
        vjepa_features: torch.Tensor
    ) -> Tuple[VJEPARecursiveCarry_Inner, torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Forward pass through recursive reasoning.

        Args:
            carry: Inner carry state (z_H, z_L, query)
            vjepa_features: V-JEPA 2 encoder features (B, S, 1024)

        Returns:
            new_carry: Updated carry state
            logits: Classification logits (B, num_classes)
            (q_halt, q_continue): ACT Q-values
        """
        seq_info = dict(
            cos_sin=self.rotary_emb() if hasattr(self, "rotary_emb") else None,
        )

        # Project V-JEPA features
        input_embeddings = self._project_input(vjepa_features)

        # Extract states
        z_H, z_L, query = carry.z_H, carry.z_L, carry.query

        # H_cycles-1 without grad (same pattern as TRM)
        with torch.no_grad():
            for _H_step in range(self.config.H_cycles - 1):
                for _L_step in range(self.config.L_cycles):
                    z_L = self.L_level(z_L, z_H + input_embeddings, **seq_info)
                z_H = self.L_level(z_H, z_L, **seq_info)

                # Update query based on first position of z_H
                query = query + self.query_update(z_H[:, 0:1])

        # Final iteration with gradients
        for _L_step in range(self.config.L_cycles):
            z_L = self.L_level(z_L, z_H + input_embeddings, **seq_info)
        z_H = self.L_level(z_H, z_L, **seq_info)

        # Final query update
        query = query + self.query_update(z_H[:, 0:1])

        # Cross-attention pooling: query attends over V-JEPA features
        pooled = self.cross_attention(query, vjepa_features)  # (B, hidden_size)

        # Classification output
        logits = self.classifier(pooled)  # (B, num_classes)

        # ACT Q-head (uses first position of z_H, same as TRM)
        q_logits = self.q_head(z_H[:, 0]).to(torch.float32)

        # New carry (detached)
        new_carry = VJEPARecursiveCarry_Inner(
            z_H=z_H.detach(),
            z_L=z_L.detach(),
            query=query.detach()
        )

        return new_carry, logits, (q_logits[..., 0], q_logits[..., 1])


class VJEPARecursiveClassifier_ACTV1(nn.Module):
    """
    ACT wrapper for V-JEPA 2 recursive video classifier.

    Implements Adaptive Computation Time: the model learns when to stop
    reasoning based on Q-values that predict classification correctness.
    """

    def __init__(self, config_dict: dict):
        super().__init__()
        self.config = VJEPARecursiveConfig(**config_dict)
        self.inner = VJEPARecursiveClassifier_ACTV1_Inner(self.config)

    def initial_carry(
        self,
        batch_size: int,
        seq_len: int,
        vjepa_features: torch.Tensor,
        labels: torch.Tensor
    ) -> VJEPARecursiveCarry:
        """Create initial carry state for a batch."""
        return VJEPARecursiveCarry(
            inner_carry=self.inner.empty_carry(batch_size, seq_len),

            steps=torch.zeros((batch_size,), dtype=torch.int32),
            halted=torch.ones((batch_size,), dtype=torch.bool),  # Start halted, will be reset

            vjepa_features=vjepa_features,
            labels=labels,
        )

    def forward(
        self,
        carry: VJEPARecursiveCarry,
        vjepa_features: torch.Tensor,
        labels: torch.Tensor
    ) -> Tuple[VJEPARecursiveCarry, Dict[str, torch.Tensor]]:
        """
        Forward pass with ACT control.

        Args:
            carry: Full carry state
            vjepa_features: V-JEPA 2 features (B, S, 1024)
            labels: Ground truth labels (B,)

        Returns:
            new_carry: Updated carry state
            outputs: Dict with logits, q_halt_logits, q_continue_logits, preds
        """
        batch_size, seq_len, _ = vjepa_features.shape

        # Reset carry for halted sequences (starting fresh)
        new_inner_carry = self.inner.reset_carry(
            carry.halted,
            carry.inner_carry,
            seq_len
        )

        new_steps = torch.where(carry.halted, 0, carry.steps)

        # Update cached features and labels for newly reset sequences
        new_vjepa_features = torch.where(
            carry.halted.view(-1, 1, 1),
            vjepa_features,
            carry.vjepa_features
        )
        new_labels = torch.where(carry.halted, labels, carry.labels)

        # Forward through inner model
        new_inner_carry, logits, (q_halt_logits, q_continue_logits) = self.inner(
            new_inner_carry,
            new_vjepa_features
        )

        outputs = {
            "logits": logits,
            "q_halt_logits": q_halt_logits,
            "q_continue_logits": q_continue_logits,
        }

        with torch.no_grad():
            # Step counter
            new_steps = new_steps + 1
            is_last_step = new_steps >= self.config.halt_max_steps

            halted = is_last_step

            # ACT halting logic during training
            if self.training and (self.config.halt_max_steps > 1):
                if self.config.no_ACT_continue:
                    halted = halted | (q_halt_logits > 0)
                else:
                    halted = halted | (q_halt_logits > q_continue_logits)

                # Exploration: randomly force more steps
                min_halt_steps = (
                    (torch.rand_like(q_halt_logits) < self.config.halt_exploration_prob) *
                    torch.randint_like(new_steps, low=2, high=self.config.halt_max_steps + 1)
                )
                halted = halted & (new_steps >= min_halt_steps)

        new_carry = VJEPARecursiveCarry(
            inner_carry=new_inner_carry,
            steps=new_steps,
            halted=halted,
            vjepa_features=new_vjepa_features,
            labels=new_labels,
        )

        return new_carry, outputs
