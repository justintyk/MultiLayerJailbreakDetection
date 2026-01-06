"""
Module 1: Data & Tasks - Generate multi-layer jailbreak dataset.

This module defines the data schema and synthetic generators for our
multi-layer activation-level jailbreak detection task.

The key idea is that each example contains:
    - activations_by_layer: dict[layer_idx -> tensor-like list of floats]
    - label:                "JAILBREAK" or "SAFE"
    - meta:                 optional metadata (prompt, notes, etc.)

In later stages, activations_by_layer will be populated from a real base LLM
via forward hooks. Here we start with a structured synthetic version that
already matches the shape our models expect.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Dict, List, Literal, TypedDict
import random


Label = Literal["JAILBREAK", "SAFE"]


class ExampleDict(TypedDict):
    activations_by_layer: Dict[int, List[List[float]]]
    label: Label
    meta: Dict[str, str]


@dataclass
class MultiLayerExample:
    """
    Structured representation of one dataset example.

    - activations_by_layer: maps each layer index (e.g., 7, 12, 23) to a
      list of activation vectors (each vector is a list[float] placeholder
      here; later this will be tensors from a real model).
    - label: "JAILBREAK" or "SAFE".
    - meta:  optional metadata such as the underlying prompt or notes.
    """
    activations_by_layer: Dict[int, List[List[float]]]
    label: Label
    meta: Dict[str, str]

    def to_dict(self) -> ExampleDict:
        return {
            "activations_by_layer": self.activations_by_layer,
            "label": self.label,
            "meta": self.meta,
        }


def _synthetic_activation_vector(dim: int = 4, bias: float = 0.0) -> List[float]:
    """
    Create a simple synthetic activation vector.

    For now this is just random noise with an optional bias; in the real
    system this will be replaced by actual hidden states from the base LLM.
    """
    return [random.uniform(-1.0, 1.0) + bias for _ in range(dim)]


def _generate_synthetic_example(
    layer_indices: List[int],
    dim: int = 4,
    jailbroken: bool = True,
) -> MultiLayerExample:
    """
    Generate one synthetic multi-layer example.

    Intuition:
    - JAILBREAK examples are biased toward a particular activation pattern
      across layers (e.g., slightly shifted means).
    - SAFE examples are drawn from a different, less aligned distribution.

    This is just a stand-in until we wire in real activations from a base model.
    """
    activations_by_layer: Dict[int, List[List[float]]] = {}
    for layer in layer_indices:
        # For now, we will be using a single position per layer; later we can use multiple.
        if jailbroken:
            # Bias jailbreak activations slightly in positive direction
            activations_by_layer[layer] = [
                _synthetic_activation_vector(dim=dim, bias=+0.5)
            ]
        else:
            # Bias safe activations slightly in negative direction
            activations_by_layer[layer] = [
                _synthetic_activation_vector(dim=dim, bias=-0.5)
            ]

    label: Label = "JAILBREAK" if jailbroken else "SAFE"
    meta = {
        "description": "synthetic multi-layer example",
        "jailbreak": str(jailbroken),
    }

    return MultiLayerExample(
        activations_by_layer=activations_by_layer,
        label=label,
        meta=meta,
    )


def generate_synthetic_multilayer_dataset(
    n_examples: int = 1000,
    layer_indices: List[int] | None = None,
    dim: int = 4,
    p_jailbreak: float = 0.5,
) -> List[ExampleDict]:
    """
    Generate a synthetic dataset of multi-layer activation examples.

    Args:
        n_examples:   total number of examples.
        layer_indices: which layers (by index) to include, e.g. [7, 12, 23].
        dim:          dimensionality of each activation vector.
        p_jailbreak:  fraction of examples that should be labeled JAILBREAK.

    Returns:
        List of examples in dict form, ready to be serialized or fed to
        a DataLoader / HF datasets.
    """
    if layer_indices is None:
        layer_indices = [7, 12, 23]

    dataset: List[ExampleDict] = []
    for _ in range(n_examples):
        jailbroken = random.random() < p_jailbreak
        ex = _generate_synthetic_example(
            layer_indices=layer_indices,
            dim=dim,
            jailbroken=jailbroken,
        )
        dataset.append(ex.to_dict())
    return dataset


if __name__ == "__main__":
    random.seed(0)
    data = generate_synthetic_multilayer_dataset(n_examples=10)
    print(f"Module 1: generated {len(data)} synthetic multi-layer examples.")
    print("Example[0]:", data[0])
