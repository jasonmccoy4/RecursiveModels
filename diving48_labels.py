"""
Diving48 Label Decomposition Utilities

Diving48 has 48 dive classes that can be decomposed into 4 components:
1. Position/Takeoff (6 types): Forward, Back, Reverse, Inward, Twist, Armstand
2. Somersault count (varies)
3. Twist count (varies)
4. Body position (4 types): Straight(A), Pike(B), Tuck(C), Free(D)

This module provides utilities to:
- Parse dive codes (e.g., "5132D") into components
- Map between composite labels and component labels
- Build component label tensors for multi-head classification
"""

import json
import re
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import torch


# Component class counts
NUM_POSITIONS = 6  # Forward, Back, Reverse, Inward, Twist, Armstand
NUM_BODY_POSITIONS = 4  # A=Straight, B=Pike, C=Tuck, D=Free

# Position mapping (first digit of dive code)
POSITION_MAP = {
    '1': 0,  # Forward
    '2': 1,  # Back
    '3': 2,  # Reverse
    '4': 3,  # Inward
    '5': 4,  # Twist
    '6': 5,  # Armstand
}

# Body position mapping (letter at end)
BODY_POSITION_MAP = {
    'A': 0,  # Straight
    'B': 1,  # Pike
    'C': 2,  # Tuck
    'D': 3,  # Free
}


@dataclass
class Diving48Components:
    """Component counts for Diving48 multi-head classification."""
    num_positions: int = NUM_POSITIONS
    num_somersaults: int = 0  # Determined from vocab
    num_twists: int = 0  # Determined from vocab
    num_body_positions: int = NUM_BODY_POSITIONS


def parse_dive_code(code: str) -> Tuple[int, float, float, int]:
    """
    Parse a dive code into its components.

    Dive codes follow the pattern: [position][somersaults][twists][body]
    Examples:
        - "101B" = Forward, 0.5 somersault, 0 twists, Pike
        - "5132D" = Twist, 1.5 somersaults, 1 twist, Free
        - "612B" = Armstand, 1 somersault, 0 twists, Pike

    Args:
        code: Dive code string (e.g., "5132D")

    Returns:
        Tuple of (position_idx, somersault_count, twist_count, body_idx)
    """
    code = code.strip().upper()

    # Extract body position (last character)
    body_char = code[-1]
    if body_char not in BODY_POSITION_MAP:
        raise ValueError(f"Invalid body position '{body_char}' in code '{code}'")
    body_idx = BODY_POSITION_MAP[body_char]

    # Extract numeric part
    numeric = code[:-1]
    if len(numeric) < 3:
        raise ValueError(f"Invalid dive code format: '{code}'")

    # Position is first digit
    position_char = numeric[0]
    if position_char not in POSITION_MAP:
        raise ValueError(f"Invalid position '{position_char}' in code '{code}'")
    position_idx = POSITION_MAP[position_char]

    # For twist dives (position 5), format is different
    if position_char == '5':
        # Twist dives: 5[somersaults][twists] e.g., 5132 = 1.5 som, 1 twist
        if len(numeric) == 4:
            somersaults = int(numeric[1]) + 0.5 * int(numeric[2])
            twists = int(numeric[3]) * 0.5
        elif len(numeric) == 3:
            somersaults = int(numeric[1]) * 0.5
            twists = int(numeric[2]) * 0.5
        else:
            somersaults = float(numeric[1:-1]) * 0.5 if len(numeric) > 2 else 0.5
            twists = int(numeric[-1]) * 0.5
    else:
        # Standard dives: [pos][01-12][optional twist]
        # Second+third digits encode somersaults (01=0.5, 02=1, etc.)
        som_digits = numeric[1:3]
        somersaults = int(som_digits) * 0.5

        # Optional twist count (fourth digit if present)
        if len(numeric) > 3:
            twists = int(numeric[3]) * 0.5
        else:
            twists = 0.0

    return position_idx, somersaults, twists, body_idx


class Diving48LabelParser:
    """
    Parser for Diving48 labels that builds component mappings.

    Given a vocab file (mapping class indices to dive codes), builds:
    - Mapping from class index to component indices
    - Lists of unique values for each component
    """

    def __init__(self, vocab_path: Optional[str] = None, vocab_dict: Optional[Dict] = None):
        """
        Initialize parser from vocab file or dict.

        Args:
            vocab_path: Path to Diving48_vocab.json
            vocab_dict: Pre-loaded vocab dictionary
        """
        if vocab_path is not None:
            with open(vocab_path, "r") as f:
                self.vocab = json.load(f)
        elif vocab_dict is not None:
            self.vocab = vocab_dict
        else:
            # Use default Diving48 V2 vocab (48 classes)
            self.vocab = self._get_default_vocab()

        self._build_mappings()

    def _get_default_vocab(self) -> Dict[str, str]:
        """Return default Diving48 V2 vocabulary."""
        # Standard Diving48 V2 vocab (48 classes)
        # Format: {class_index: dive_code}
        return {
            "0": "101B", "1": "101C", "2": "103B", "3": "103C",
            "4": "105B", "5": "105C", "6": "107C", "7": "109C",
            "8": "201B", "9": "201C", "10": "203B", "11": "203C",
            "12": "205B", "13": "205C", "14": "207C", "15": "301B",
            "16": "301C", "17": "303B", "18": "303C", "19": "305B",
            "20": "305C", "21": "307C", "22": "401B", "23": "401C",
            "24": "403B", "25": "403C", "26": "405B", "27": "405C",
            "28": "407C", "29": "5122D", "30": "5124D", "31": "5126D",
            "32": "5132D", "33": "5134D", "34": "5136D", "35": "5152D",
            "36": "5154D", "37": "5231D", "38": "5233D", "39": "5235D",
            "40": "5251D", "41": "5253D", "42": "5255D", "43": "612B",
            "44": "614B", "45": "624C", "46": "626C", "47": "628C",
        }

    def _build_mappings(self):
        """Build component mappings from vocab."""
        self.class_to_components: Dict[int, Tuple[int, float, float, int]] = {}

        somersault_values = set()
        twist_values = set()

        # Parse all dive codes
        for class_idx_str, code in self.vocab.items():
            class_idx = int(class_idx_str)
            pos, som, twist, body = parse_dive_code(code)
            self.class_to_components[class_idx] = (pos, som, twist, body)
            somersault_values.add(som)
            twist_values.add(twist)

        # Build ordered lists of unique values
        self.somersault_values = sorted(somersault_values)
        self.twist_values = sorted(twist_values)

        # Build value to index mappings
        self.somersault_to_idx = {v: i for i, v in enumerate(self.somersault_values)}
        self.twist_to_idx = {v: i for i, v in enumerate(self.twist_values)}

        # Component info
        self.components = Diving48Components(
            num_positions=NUM_POSITIONS,
            num_somersaults=len(self.somersault_values),
            num_twists=len(self.twist_values),
            num_body_positions=NUM_BODY_POSITIONS,
        )

        # Build class to component index mappings
        self.class_to_component_indices: Dict[int, Tuple[int, int, int, int]] = {}
        for class_idx, (pos, som, twist, body) in self.class_to_components.items():
            som_idx = self.somersault_to_idx[som]
            twist_idx = self.twist_to_idx[twist]
            self.class_to_component_indices[class_idx] = (pos, som_idx, twist_idx, body)

    def get_component_labels(self, class_label: int) -> Tuple[int, int, int, int]:
        """
        Get component indices for a class label.

        Args:
            class_label: Class index (0-47)

        Returns:
            Tuple of (position_idx, somersault_idx, twist_idx, body_idx)
        """
        return self.class_to_component_indices[class_label]

    def get_component_labels_batch(self, class_labels: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Get component labels for a batch.

        Args:
            class_labels: Tensor of class indices (B,)

        Returns:
            Dict with 'position', 'somersault', 'twist', 'body' tensors
        """
        device = class_labels.device
        batch_size = class_labels.shape[0]

        positions = torch.zeros(batch_size, dtype=torch.long, device=device)
        somersaults = torch.zeros(batch_size, dtype=torch.long, device=device)
        twists = torch.zeros(batch_size, dtype=torch.long, device=device)
        bodies = torch.zeros(batch_size, dtype=torch.long, device=device)

        for i, label in enumerate(class_labels.tolist()):
            pos, som, twist, body = self.class_to_component_indices[label]
            positions[i] = pos
            somersaults[i] = som
            twists[i] = twist
            bodies[i] = body

        return {
            'position': positions,
            'somersault': somersaults,
            'twist': twists,
            'body': bodies,
        }

    def build_component_lookup_tensors(self, device: torch.device = None) -> Dict[str, torch.Tensor]:
        """
        Build lookup tensors for fast batch component extraction.

        Returns:
            Dict with lookup tensors for each component
        """
        num_classes = len(self.vocab)
        position_lookup = torch.zeros(num_classes, dtype=torch.long, device=device)
        somersault_lookup = torch.zeros(num_classes, dtype=torch.long, device=device)
        twist_lookup = torch.zeros(num_classes, dtype=torch.long, device=device)
        body_lookup = torch.zeros(num_classes, dtype=torch.long, device=device)

        for class_idx, (pos, som, twist, body) in self.class_to_component_indices.items():
            position_lookup[class_idx] = pos
            somersault_lookup[class_idx] = som
            twist_lookup[class_idx] = twist
            body_lookup[class_idx] = body

        return {
            'position': position_lookup,
            'somersault': somersault_lookup,
            'twist': twist_lookup,
            'body': body_lookup,
        }


# Global parser instance (lazy-loaded)
_default_parser: Optional[Diving48LabelParser] = None


def get_default_parser() -> Diving48LabelParser:
    """Get the default Diving48 label parser."""
    global _default_parser
    if _default_parser is None:
        _default_parser = Diving48LabelParser()
    return _default_parser


def get_component_counts() -> Diving48Components:
    """Get component class counts for Diving48."""
    return get_default_parser().components
