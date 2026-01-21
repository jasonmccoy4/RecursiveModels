"""
V-JEPA 2 Recursive Video Classifier with Scalar Gated Feature Grouping

Like trm_vjepa_gated but with a 3-level recursive structure (H, L, G).
z_G acts as an element-wise gate on grouped VJEPA features.

Architecture:
1. Input: V-JEPA 2 features (1024-dim)
2. Feature grouping: 1024 -> hidden_size groups (e.g., 256 groups of 4)
3. z_G gates each group via sigmoid
4. 3-level recursion: H_cycles > G_cycles > L_cycles
"""

from typing import Tuple, Dict
from dataclasses import dataclass
import torch
import torch.nn.functional as F
from torch import nn
from pydantic import BaseModel

from models.common import trunc_normal_init_
from models.layers import (
    rms_norm, SwiGLU, Attention, RotaryEmbedding3D, CosSin,
    CastedLinear
)
from diving48_labels import get_component_counts


@dataclass
class VJEPARecursiveCarry_Inner:
    """Inner carry state for recursive reasoning."""
    z_H: torch.Tensor  # High-level reasoning state: (B, S, D)
    z_L: torch.Tensor  # Low-level reasoning state: (B, S, D)
    z_G: torch.Tensor  # Convolutional gate state: (B, S, D)


@dataclass
class VJEPARecursiveCarry:
    """Full carry state including ACT control."""
    inner_carry: VJEPARecursiveCarry_Inner

    steps: torch.Tensor  # Step counter per sample: (B,)
    halted: torch.Tensor  # Halt flag per sample: (B,)

    vjepa_features: torch.Tensor  # Cached V-JEPA features: (B, S, D_vjepa)
    labels: torch.Tensor  # Ground truth labels: (B,)


class VJEPARecursiveConfig(BaseModel):
    """Configuration for V-JEPA 2 Recursive Classifier with Conv Gating."""

    batch_size: int

    # V-JEPA 2 config
    vjepa_hidden_size: int = 1024  # V-JEPA 2 ViT-L output dimension

    # Recursive model config - original TRM dimensions
    hidden_size: int = 512  # Original TRM dimension
    H_cycles: int = 2
    L_cycles: int = 4
    G_cycles: int = 3
    L_layers: int = 2

    # Transformer config - original TRM
    expansion: float = 4.0
    num_heads: int = 4  # Original TRM: 4 heads, head_dim=128
    pos_encodings: str = "rope"
    rms_norm_eps: float = 1e-5
    rope_theta: float = 10000.0

    # ACT config
    halt_max_steps: int = 8
    halt_exploration_prob: float = 0.1
    no_ACT_continue: bool = True

    # Classification config
    num_classes: int = 174  # Something-Something V2

    # 3D-RoPE config for V-JEPA spatial-temporal structure
    grid_size: Tuple[int, int, int] = (32, 16, 16)  # (T, H, W) for V-JEPA 2

    # Gating config
    interpolated_gating: bool = False  # Upsample z_G to full resolution for softer boundaries

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


class ScalarGatedFeatureGrouping(nn.Module):
    """
    Scalar gated feature grouping - no learned projection.

    Groups VJEPA features into hidden_size groups, sums each group,
    then applies z_G as a scalar gate per group.

    Example: 1024 features -> 256 groups of 4 -> sum each group -> gate by z_G
    """

    def __init__(self, config: VJEPARecursiveConfig) -> None:
        super().__init__()
        self.config = config
        self.group_size = config.vjepa_hidden_size // config.hidden_size  # e.g., 1024/256 = 4

    def forward(self, vjepa_features: torch.Tensor, z_G: torch.Tensor) -> torch.Tensor:
        """
        Apply scalar gated feature grouping.

        Args:
            vjepa_features: (B, S, 1024) V-JEPA 2 features
            z_G: (B, S, hidden_size) gating state (one scalar per group)

        Returns:
            output: (B, S, hidden_size) grouped and gated features
        """
        B, S, D = vjepa_features.shape

        if self.config.interpolated_gating:
            # Interpolated gating: upsample z_G to full resolution for softer boundaries
            # (B, S, hidden_size) -> (B*S, 1, hidden_size) -> interpolate -> (B*S, 1, D) -> (B, S, D)
            z_G_flat = z_G.view(B * S, 1, -1)  # (B*S, 1, 256)
            z_G_full = F.interpolate(
                z_G_flat,
                size=D,
                mode='linear',
                align_corners=True
            ).view(B, S, D)  # (B, S, 1024)

            # Apply per-feature gating
            gate = torch.sigmoid(z_G_full)
            x = vjepa_features * gate  # (B, S, D)

            # Sum groups after gating
            x = x.view(B, S, -1, self.group_size)  # (B, S, 256, 4)
            return x.sum(dim=-1)  # (B, S, 256)
        else:
            # Original: hard group boundaries
            x = vjepa_features.view(B, S, -1, self.group_size)  # (B, S, 256, 4)
            x = x.sum(dim=-1)  # (B, S, 256)
            gate = torch.sigmoid(z_G)
            return x * gate


class VJEPARecursiveClassifier_ACTV1_Inner(nn.Module):
    """
    Inner model for V-JEPA 2 recursive video classification with scalar gated feature grouping.

    Takes V-JEPA 2 features and performs recursive reasoning to produce
    classification logits and ACT halting decisions.
    """

    def __init__(self, config: VJEPARecursiveConfig) -> None:
        super().__init__()
        self.config = config
        self.forward_dtype = getattr(torch, self.config.forward_dtype)

        # 3D Position encodings for video features (T, H, W structure)
        if config.pos_encodings == "rope":
            self.rotary_emb = RotaryEmbedding3D(
                dim=config.hidden_size // config.num_heads,
                grid_size=config.grid_size,  # (T, H, W) = (32, 16, 16) for V-JEPA 2
                base=config.rope_theta
            )

        # Scalar gated feature grouping (groups input features, z_G gates each group)
        self.gated_proj = ScalarGatedFeatureGrouping(config)

        # Recursive reasoning layers (L-level, same structure as TRM)
        self.L_level = VJEPARecursiveReasoningModule(
            layers=[VJEPARecursiveBlock(config) for _ in range(config.L_layers)]
        )

        # Initial states for H, L, and G
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
        self.G_init = nn.Buffer(
            trunc_normal_init_(
                torch.empty(config.hidden_size, dtype=self.forward_dtype),
                std=1
            ),
            persistent=True
        )

        # Classification head (replaces lm_head)
        self.classifier = CastedLinear(config.hidden_size, config.num_classes, bias=True)

        # ACT Q-head (same as TRM)
        self.q_head = CastedLinear(config.hidden_size, 2, bias=True)

        # Q-head initialization (same as TRM)
        with torch.no_grad():
            self.q_head.weight.zero_()
            self.q_head.bias.fill_(-5)

    def empty_carry(self, batch_size: int, seq_len: int, device: torch.device = None) -> VJEPARecursiveCarry_Inner:
        """Create empty inner carry state."""
        if device is None:
            device = next(self.parameters()).device
        return VJEPARecursiveCarry_Inner(
            z_H=torch.empty(
                batch_size, seq_len, self.config.hidden_size,
                dtype=self.forward_dtype, device=device
            ),
            z_L=torch.empty(
                batch_size, seq_len, self.config.hidden_size,
                dtype=self.forward_dtype, device=device
            ),
            z_G=torch.empty(
                batch_size, seq_len, self.config.hidden_size,
                dtype=self.forward_dtype, device=device
            ),
        )

    def reset_carry(
        self,
        reset_flag: torch.Tensor,
        carry: VJEPARecursiveCarry_Inner,
        seq_len: int,
        input_embeddings: torch.Tensor = None  # Unused (kept for API compatibility)
    ) -> VJEPARecursiveCarry_Inner:
        """Reset carry state for halted sequences."""
        del input_embeddings  # Explicitly mark as unused
        batch_size = reset_flag.shape[0]

        # Expand init states to sequence length
        H_init_expanded = self.H_init.unsqueeze(0).unsqueeze(0).expand(batch_size, seq_len, -1)
        L_init_expanded = self.L_init.unsqueeze(0).unsqueeze(0).expand(batch_size, seq_len, -1)
        G_init_expanded = self.G_init.unsqueeze(0).unsqueeze(0).expand(batch_size, seq_len, -1)

        return VJEPARecursiveCarry_Inner(
            z_H=torch.where(reset_flag.view(-1, 1, 1), H_init_expanded, carry.z_H),
            z_L=torch.where(reset_flag.view(-1, 1, 1), L_init_expanded, carry.z_L),
            z_G=torch.where(reset_flag.view(-1, 1, 1), G_init_expanded, carry.z_G),
        )

    def forward(
        self,
        carry: VJEPARecursiveCarry_Inner,
        vjepa_features: torch.Tensor
    ) -> Tuple[VJEPARecursiveCarry_Inner, torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Forward pass through recursive reasoning with 3-level structure.

        z_G gates the grouped VJEPA features (1024 -> hidden_size).

        Structure:
        - L_cycles (inner): z_L updates with gated input injection
        - G_cycles (middle): z_G (gate state) updates based on z_L
        - H_cycles (outer): z_H updates based on z_L

        Args:
            carry: Inner carry state (z_H, z_L, z_G)
            vjepa_features: V-JEPA 2 encoder features (B, S, 1024)

        Returns:
            new_carry: Updated carry state
            logits: Classification logits (B, num_classes)
            (q_halt, q_continue): ACT Q-values
        """
        seq_info = dict(
            cos_sin=self.rotary_emb() if hasattr(self, "rotary_emb") else None,
        )

        # Extract states
        z_H, z_L, z_G = carry.z_H, carry.z_L, carry.z_G

        def get_injection():
            # z_G gates the grouped VJEPA features
            projected_input = self.gated_proj(vjepa_features, z_G)
            return z_H + projected_input

        # H_cycles-1 without grad (same pattern as TRM)
        with torch.no_grad():
            for _H_step in range(self.config.H_cycles - 1):
                for _G_step in range(self.config.G_cycles):
                    for _L_step in range(self.config.L_cycles):
                        z_L = self.L_level(z_L, get_injection(), **seq_info)
                    z_G = self.L_level(z_G, z_L, **seq_info)
                z_H = self.L_level(z_H, z_L, **seq_info)

        # Final iteration with gradients
        for _G_step in range(self.config.G_cycles):
            for _L_step in range(self.config.L_cycles):
                z_L = self.L_level(z_L, get_injection(), **seq_info)
            z_G = self.L_level(z_G, z_L, **seq_info)
        z_H = self.L_level(z_H, z_L, **seq_info)

        # Mean pooling for classification (aggregate all spatial-temporal info)
        pooled = z_H.mean(dim=1)  # (B, D)
        logits = self.classifier(pooled)  # (B, num_classes)

        # ACT Q-head
        q_logits = self.q_head(pooled).to(torch.float32)

        # New carry (detached)
        new_carry = VJEPARecursiveCarry_Inner(
            z_H=z_H.detach(),
            z_L=z_L.detach(),
            z_G=z_G.detach(),
        )

        return new_carry, logits, (q_logits[..., 0], q_logits[..., 1])


class VJEPARecursiveClassifier_ACTV1(nn.Module):
    """
    ACT wrapper for V-JEPA 2 recursive video classifier with scalar gated feature grouping.

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
        device = vjepa_features.device
        return VJEPARecursiveCarry(
            inner_carry=self.inner.empty_carry(batch_size, seq_len, device=device),

            steps=torch.zeros((batch_size,), dtype=torch.int32, device=device),
            halted=torch.ones((batch_size,), dtype=torch.bool, device=device),  # Start halted, will be reset

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
            seq_len,
            input_embeddings=vjepa_features
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


class VJEPARecursiveClassifier_MultiHead_Inner(VJEPARecursiveClassifier_ACTV1_Inner):
    """
    Multi-head variant for Diving48 classification with scalar gated feature grouping.

    Outputs separate logits for each component:
    - Position (6 classes)
    - Somersault count (determined from vocab)
    - Twist count (determined from vocab)
    - Body position (4 classes)
    """

    def __init__(self, config: VJEPARecursiveConfig) -> None:
        # Initialize parent but we'll replace the classifier
        super().__init__(config)

        # Get component counts
        components = get_component_counts()

        # Replace single classifier with multi-head classifiers
        del self.classifier

        self.position_head = CastedLinear(config.hidden_size, components.num_positions, bias=True)
        self.somersault_head = CastedLinear(config.hidden_size, components.num_somersaults, bias=True)
        self.twist_head = CastedLinear(config.hidden_size, components.num_twists, bias=True)
        self.body_head = CastedLinear(config.hidden_size, components.num_body_positions, bias=True)

        # Store component counts
        self.num_positions = components.num_positions
        self.num_somersaults = components.num_somersaults
        self.num_twists = components.num_twists
        self.num_body_positions = components.num_body_positions

    def forward(
        self,
        carry: VJEPARecursiveCarry_Inner,
        vjepa_features: torch.Tensor
    ) -> Tuple[VJEPARecursiveCarry_Inner, Dict[str, torch.Tensor], Tuple[torch.Tensor, torch.Tensor]]:
        """
        Forward pass with multi-head output and scalar gated feature grouping.

        Structure:
        - L_cycles (inner): z_L updates with gated input injection
        - G_cycles (middle): z_G (gate state) updates based on z_L
        - H_cycles (outer): z_H updates based on z_L

        Returns:
            new_carry: Updated carry state
            logits_dict: Dict with position/somersault/twist/body logits
            (q_halt, q_continue): ACT Q-values
        """
        seq_info = dict(
            cos_sin=self.rotary_emb() if hasattr(self, "rotary_emb") else None,
        )

        z_H, z_L, z_G = carry.z_H, carry.z_L, carry.z_G

        def get_injection():
            # z_G gates the grouped VJEPA features
            projected_input = self.gated_proj(vjepa_features, z_G)
            return z_H + projected_input

        # H_cycles-1 without grad
        with torch.no_grad():
            for _H_step in range(self.config.H_cycles - 1):
                for _G_step in range(self.config.G_cycles):
                    for _L_step in range(self.config.L_cycles):
                        z_L = self.L_level(z_L, get_injection(), **seq_info)
                    z_G = self.L_level(z_G, z_L, **seq_info)
                z_H = self.L_level(z_H, z_L, **seq_info)

        # Final iteration with gradients
        for _G_step in range(self.config.G_cycles):
            for _L_step in range(self.config.L_cycles):
                z_L = self.L_level(z_L, get_injection(), **seq_info)
            z_G = self.L_level(z_G, z_L, **seq_info)
        z_H = self.L_level(z_H, z_L, **seq_info)

        # Mean pooling for multi-head classification
        hidden = z_H.mean(dim=1)  # (B, D)
        position_logits = self.position_head(hidden)
        somersault_logits = self.somersault_head(hidden)
        twist_logits = self.twist_head(hidden)
        body_logits = self.body_head(hidden)

        # ACT Q-head
        q_logits = self.q_head(hidden).to(torch.float32)

        new_carry = VJEPARecursiveCarry_Inner(
            z_H=z_H.detach(),
            z_L=z_L.detach(),
            z_G=z_G.detach(),
        )

        return new_carry, {
            "position_logits": position_logits,
            "somersault_logits": somersault_logits,
            "twist_logits": twist_logits,
            "body_logits": body_logits,
        }, (q_logits[..., 0], q_logits[..., 1])


class VJEPARecursiveClassifier_MultiHead(nn.Module):
    """
    ACT wrapper for multi-head Diving48 classifier with scalar gated feature grouping.

    Supports both multi-head and single-head modes via head_type config.
    """

    def __init__(self, config_dict: dict):
        super().__init__()
        self.config = VJEPARecursiveConfig(**config_dict)
        self.head_type = config_dict.get("head_type", "multi")

        if self.head_type == "multi":
            self.inner = VJEPARecursiveClassifier_MultiHead_Inner(self.config)
        else:
            self.inner = VJEPARecursiveClassifier_ACTV1_Inner(self.config)

    def initial_carry(
        self,
        batch_size: int,
        seq_len: int,
        vjepa_features: torch.Tensor,
        labels: torch.Tensor
    ) -> VJEPARecursiveCarry:
        """Create initial carry state for a batch."""
        device = vjepa_features.device
        return VJEPARecursiveCarry(
            inner_carry=self.inner.empty_carry(batch_size, seq_len, device=device),
            steps=torch.zeros((batch_size,), dtype=torch.int32, device=device),
            halted=torch.ones((batch_size,), dtype=torch.bool, device=device),
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
        """
        batch_size, seq_len, _ = vjepa_features.shape

        # Reset carry for halted sequences (starting fresh)
        new_inner_carry = self.inner.reset_carry(
            carry.halted,
            carry.inner_carry,
            seq_len,
            input_embeddings=vjepa_features
        )

        new_steps = torch.where(carry.halted, 0, carry.steps)

        new_vjepa_features = torch.where(
            carry.halted.view(-1, 1, 1),
            vjepa_features,
            carry.vjepa_features
        )
        new_labels = torch.where(carry.halted, labels, carry.labels)

        if self.head_type == "multi":
            new_inner_carry, logits_dict, (q_halt_logits, q_continue_logits) = self.inner(
                new_inner_carry,
                new_vjepa_features
            )
            outputs = {
                **logits_dict,
                "q_halt_logits": q_halt_logits,
                "q_continue_logits": q_continue_logits,
            }
        else:
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
            new_steps = new_steps + 1
            is_last_step = new_steps >= self.config.halt_max_steps

            halted = is_last_step

            if self.training and (self.config.halt_max_steps > 1):
                if self.config.no_ACT_continue:
                    halted = halted | (q_halt_logits > 0)
                else:
                    halted = halted | (q_halt_logits > q_continue_logits)

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
