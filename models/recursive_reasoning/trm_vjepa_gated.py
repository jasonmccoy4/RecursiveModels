"""
V-JEPA 2 Recursive Video Classifier

Adapts the TinyRecursiveReasoningModel (TRM) architecture for video classification
using V-JEPA 2 as a frozen feature extractor. Key changes from original TRM:

1. Input: V-JEPA 2 features (1024-dim) instead of token embeddings
2. Hidden size: 1024 (matched to V-JEPA 2)
3. Output: Classification head (174 classes) instead of LM head
4. Pooling: Mean pooling over full V-JEPA sequence for classification
5. ACT: Retained for adaptive computation
6. Position encoding: 3D-RoPE for spatial-temporal structure (T, H, W)
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
    z_G: torch.Tensor  # Gated-level reasoning state: (B, S, D)


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
    H_cycles: int = 2
    L_cycles: int = 4
    G_cycles: int = 3
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

    # Cross-attention config (legacy, unused)
    cross_attn_heads: int = 8

    # 3D-RoPE config for V-JEPA spatial-temporal structure
    grid_size: Tuple[int, int, int] = (32, 16, 16)  # (T, H, W) for V-JEPA 2

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

        # 3D Position encodings for video features (T, H, W structure)
        if config.pos_encodings == "rope":
            self.rotary_emb = RotaryEmbedding3D(
                dim=config.hidden_size // config.num_heads,
                grid_size=config.grid_size,  # (T, H, W) = (32, 16, 16) for V-JEPA 2
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
        input_embeddings: torch.Tensor = None  # Unused in gated version (kept for API compatibility)
    ) -> VJEPARecursiveCarry_Inner:
        """Reset carry state for halted sequences.

        Args:
            reset_flag: Boolean tensor indicating which sequences to reset
            carry: Current carry state
            seq_len: Sequence length
            input_embeddings: Unused in gated version (kept for API compatibility)
        """
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
        Forward pass through recursive reasoning with 3-level gated structure.

        Structure:
        - L_cycles (inner): z_L updates with gated input injection
        - G_cycles (middle): z_G (element-wise gate) updates based on z_L
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

        # Input embeddings
        input_embeddings = vjepa_features

        def get_injection():
            # z_G acts as element-wise gate for input injection
            gate = torch.sigmoid(z_G)
            return z_H + gate * input_embeddings

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
        # For initial_only mode, pass input_embeddings to initialize z_L
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
    Multi-head variant for Diving48 classification.

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
        Forward pass with multi-head output and 3-level gated structure.

        Structure:
        - L_cycles (inner): z_L updates with gated input injection
        - G_cycles (middle): z_G (element-wise gate) updates based on z_L
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

        # Input embeddings
        input_embeddings = vjepa_features

        def get_injection():
            # z_G acts as element-wise gate for input injection
            gate = torch.sigmoid(z_G)
            return z_H + gate * input_embeddings

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
    ACT wrapper for multi-head Diving48 classifier.

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
        # For initial_only mode, pass input_embeddings to initialize z_L
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
