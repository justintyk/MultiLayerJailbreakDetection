"""
Module 4: Evaluation - Jailbreak detection accuracy + CSV.

This module evaluates a trained activation-level classifier on a held-out
multi-layer jailbreak dataset and writes results to a CSV file.
"""

from __future__ import annotations

import csv
from typing import Dict, List, Tuple

import torch
from torch.utils.data import Dataset, DataLoader

import sys
import os

# Add the project root directory to sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

from src.data import generate_synthetic_multilayer_dataset
from src.models import MultiLayerActivationOracle
from src.train import MultiLayerJailbreakDataset, LABEL_MAP


def evaluate_model(
    model,
    n_examples: int = 200,
    batch_size: int = 32,
    device: str = "cuda",
    csv_path: str = "results.csv",
) -> Tuple[float, float]:
    """
    We need to evaluate a trained activation-level classifier on held-out data and
    write aggregate metrics to a CSV.

    Args:
        model:      Trained ActivationHead (or compatible classifier).
        n_examples: Number of held-out examples to generate.
        batch_size: Evaluation batch size.
        device:     "cuda" or "cpu".
        csv_path:   Where to save the results CSV.

    Returns:
        (accuracy, stderr_95): accuracy and 95% standard error estimate.
    """
    # 1. Generate held-out synthetic data (later: real held-out split).
    examples = generate_synthetic_multilayer_dataset(n_examples=n_examples)
    dataset = MultiLayerJailbreakDataset(examples)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    oracle = MultiLayerActivationOracle(device=device)
    model.to(device)
    model.eval()

    total_correct = 0
    total_examples = 0

    with torch.no_grad():
        for activations_by_layer, labels in dataloader:
            labels = labels.to(device)

            # Aggregate multi-layer activations exactly as in training.
            pooled_reps = []
            for i in range(len(labels)):
                acts = {layer: tensor[i].to(device) for layer, tensor in activations_by_layer.items()}
                agg_i = oracle._encode_multi_layer_activations(acts)  # [K, D]
                pooled_reps.append(agg_i.mean(dim=0))
            batch_rep = torch.stack(pooled_reps, dim=0)  # [B, D]

            logits = model(batch_rep)
            preds = logits.argmax(dim=-1)
            total_correct += (preds == labels).sum().item()
            total_examples += labels.size(0)

    accuracy = total_correct / total_examples if total_examples > 0 else 0.0
    # Simple 95% binomial standard error:
    if total_examples > 0:
        se = (accuracy * (1 - accuracy) / total_examples) ** 0.5 * 1.96
    else:
        se = 0.0

    # 2. Write summary metrics to CSV.
    with open(csv_path, mode="w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["metric", "value"])
        writer.writerow(["n_examples", total_examples])
        writer.writerow(["accuracy", f"{accuracy:.4f}"])
        writer.writerow(["stderr_95", f"{se:.4f}"])

    print(f"Module 4: Evaluation complete. Accuracy = {accuracy:.4f} ± {se:.4f}")
    print(f"Results saved to {csv_path}")
    return accuracy, se


if __name__ == "__main__":
    # Example usage: load or import a trained model from src.train.
    from src.train import train_model

    # For a quick test, train a small model on CPU, then evaluate.
    trained_model = train_model(device="cpu")
    evaluate_model(trained_model, device="cpu")
