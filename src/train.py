"""
Module 3: Training - Fine-tune Activation Oracle on multi-layer jailbreak dataset.

This module wires together:
    - the synthetic (or later real) multi-layer activation dataset from src.data
    - the Activation Oracle wrapper from src.models
    - a simple supervised training loop that optimizes a jailbreak detection objective

In a production setting, we would likely switch to TRL's SFTTrainer, but this
file shows explicitly what we are optimizing and how it connects to activations.
"""

from __future__ import annotations

from typing import Dict, List

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader

import sys
import os

# Add the project root directory to sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

from src.data import generate_synthetic_multilayer_dataset
from src.models import MultiLayerActivationOracle


LABEL_MAP: Dict[str, int] = {
    "SAFE": 0,
    "JAILBREAK": 1,
}


class MultiLayerJailbreakDataset(Dataset):
    """
    Thin Dataset wrapper around the dict-based examples from src.data.

    Each item contains:
        - activations_by_layer: dict[int -> list[list[float]]]
        - label:                int (0 = SAFE, 1 = JAILBREAK)
    """

    def __init__(self, examples: List[dict]) -> None:
        self.examples = examples

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int):
        ex = self.examples[idx]
        activations_by_layer = {
            int(layer): torch.tensor(vectors, dtype=torch.float32)
            for layer, vectors in ex["activations_by_layer"].items()
        }
        label_str = ex["label"]
        label = LABEL_MAP[label_str]
        return activations_by_layer, torch.tensor(label, dtype=torch.long)


class ActivationHead(nn.Module):
    """
    Simple classifier head that sits on top of the multi-layer activation representation.

    Instead of directly fine-tuning the full AO, we start with a small head that takes
    the aggregated activation representation (from MultiLayerActivationOracle) as input
    and predicts a binary jailbreak label. This is where the novelty in the activation
    representation is directly coupled to a supervised safety objective.
    """

    def __init__(self, input_dim: int, hidden_dim: int = 128, num_labels: int = 2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_labels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the classifier head.

        Expected shapes:
          - [B, D]    : batch of already-pooled representations
          - [B, K, D] : batch with K layers per example
          - [D]       : single example (we add a batch dim)
        """
        if x.dim() == 3: # [B, K, D] -> [B, D] (mean pool over K)
            x = x.mean(dim=1)
        elif x.dim() == 1: # [D] -> [1, D]
            x = x.unsqueeze(0)
        return self.net(x)


def train_model(
    n_examples: int = 1000,
    batch_size: int = 32,
    num_epochs: int = 3,
    device: str = "cuda",
):
    """
    Train an activation-level classifier on multi-layer jailbreak data.

    This function is deliberately transparent: it shows how multi-layer activations
    are turned into a pooled representation and optimized for the jailbreak vs
    safe objective. Later, you can extend this to:
        - train the full AO jointly,
        - use TRL's SFTTrainer for a text-generating AO,
        - or incorporate more complex architectures.
    """
    # 1. Generate synthetic dataset (later: we need to replace it with real activations).
    examples = generate_synthetic_multilayer_dataset(n_examples=n_examples)
    dataset = MultiLayerJailbreakDataset(examples)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # 2. Initialize the Activation Oracle wrapper.
    #    We use it here to define the representation dimensionality and to keep
    #    the interface aligned with src.models.
    oracle = MultiLayerActivationOracle(device=device)

    # Peek at a single example to infer activation dimension.
    sample_acts, _ = dataset[0]
    agg = oracle._encode_multi_layer_activations(sample_acts)  # [K, D]
    input_dim = agg.shape[-1]

    # 3. Initialize a classifier head on top of the aggregated activations.
    model = ActivationHead(input_dim=input_dim).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    model.train()
    for epoch in range(num_epochs):
        total_loss = 0.0
        total_correct = 0
        total_examples = 0

        for activations_by_layer, labels in dataloader:
            # Move labels to device.
            labels = labels.to(device)

            # Aggregate multi-layer activations for each example.
            # This is where our multi-layer representation is actually used.
            pooled_reps = []
            for i in range(len(labels)):
                acts = {layer: tensor[i].to(device) for layer, tensor in activations_by_layer.items()}
                agg_i = oracle._encode_multi_layer_activations(acts)  # [K, D]
                # Pool over K positions to get a single vector [D].
                pooled_reps.append(agg_i.mean(dim=0))
            batch_rep = torch.stack(pooled_reps, dim=0)  # [B, D]

            # Forward pass through the classifier head.
            logits = model(batch_rep)  # [B, 2]
            loss = criterion(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * labels.size(0)
            preds = logits.argmax(dim=-1)
            total_correct += (preds == labels).sum().item()
            total_examples += labels.size(0)

        avg_loss = total_loss / total_examples
        acc = total_correct / total_examples
        print(f"Epoch {epoch+1}/{num_epochs} - loss: {avg_loss:.4f} - acc: {acc:.4f}")

    print("Module 3: Training complete (activation-level classifier).")
    return model


if __name__ == "__main__":
    # We can start with CPU for quick tests; then we will switch to "cuda" when the GPU is available.
    trained_model = train_model(device="cpu")
