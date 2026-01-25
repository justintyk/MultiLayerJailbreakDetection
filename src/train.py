"""
Module: Policy Training - Train policy network to discover jailbreak interventions.

This module implements the training loop for the policy network, following
the investigator agent framework adapted to activation space.

Training phases:
1. Exploration: Random perturbation sampling to find seed directions
2. Policy Gradient: REINFORCE with baseline and entropy regularization
3. Logging: Track metrics and save checkpoints

Key components:
- exploration_phase(): Random search for successful interventions
- policy_gradient_phase(): Train policy with REINFORCE
- train_policy(): Main training loop
"""

from typing import List, Dict, Tuple, Optional, Any, Iterable
from datetime import datetime
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
from pathlib import Path
import json
import sys
from tqdm import tqdm

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.policy import ConceptExplainer
from src.intervention import InterventionPipeline
from src.rubrics import RubricDefinition, Verifier
from src.data import BaseModelActivationExtractor, generate_rubric_dataset


def _append_jsonl(path: Path, record: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")


def _serialize_rubric_dataset(
    dataset: Iterable[Any],
    output_path: Path
) -> None:
    serialized = []
    for item in dataset:
        if hasattr(item, "to_dict"):
            serialized.append(item.to_dict())
        elif isinstance(item, dict):
            serialized.append(item)
        else:
            raise ValueError(f"Unsupported dataset item type: {type(item)}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(serialized, f, indent=2)


def _load_rubric_dataset(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def build_policy_training_examples(
    rubric_examples: Iterable[Any],
    layer_idx: int
) -> List[Dict[str, Any]]:
    """
    Convert rubric dataset items into policy training examples.
    
    Each output example contains:
    - rubric: RubricDefinition
    - base_activation: tensor for layer_idx
    - prompt: original prompt
    """
    training_examples: List[Dict[str, Any]] = []
    for example in rubric_examples:
        if hasattr(example, "activations_by_layer"):
            activations_by_layer = example.activations_by_layer
            rubric_text = example.rubric_text
            target_answer = example.target_answer
            category = getattr(example, "category", "jailbreak")
            metadata = getattr(example, "meta", {})
            prompt = example.prompt
        elif isinstance(example, dict):
            activations_by_layer = example["activations_by_layer"]
            rubric_info = example.get("rubric", {})
            rubric_text = rubric_info.get("rubric_text")
            target_answer = rubric_info.get("target_answer")
            category = rubric_info.get("category", "jailbreak")
            metadata = example.get("meta", {})
            prompt = example["prompt"]
        else:
            raise ValueError(f"Unsupported example type: {type(example)}")
        
        if rubric_text is None or target_answer is None:
            raise ValueError("Rubric text and target answer are required for training examples.")
        
        layer_key = layer_idx
        if layer_key not in activations_by_layer and str(layer_idx) in activations_by_layer:
            layer_key = str(layer_idx)
        
        base_act = activations_by_layer[layer_key]
        if not isinstance(base_act, torch.Tensor):
            base_act = torch.tensor(base_act, dtype=torch.float32)
        if base_act.dim() > 1:
            base_act = base_act.squeeze(0)
        
        rubric = RubricDefinition(
            rubric_text=rubric_text,
            target_answer=target_answer,
            category=category,
            metadata=metadata
        )
        
        training_examples.append({
            "rubric": rubric,
            "base_activation": base_act,
            "prompt": prompt
        })
    
    return training_examples


class PolicyTrainingDataset(Dataset):
    """
    Dataset for policy training.
    
    Each example contains:
    - rubric: RubricDefinition
    - base_activation: f(x) from neutral prompt
    - prompt: Neutral prompt text
    """
    
    def __init__(self, examples: List[Dict]):
        """
        Initialize dataset.
        
        Args:
            examples: List of dicts with keys: rubric, base_activation, prompt
        """
        self.examples = examples
    
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        return self.examples[idx]


def custom_collate_fn(batch: List[Dict]) -> Dict:
    """
    Custom collate function for PolicyTrainingDataset.
    
    Handles RubricDefinition objects and other non-tensor types properly.
    """
    # Separate items that need special handling
    rubrics = [item['rubric'] for item in batch]
    prompts = [item['prompt'] for item in batch]
    
    # Handle base_activation tensors
    base_activations = []
    for item in batch:
        base_act = item['base_activation']
        # Ensure it's a tensor
        if not isinstance(base_act, torch.Tensor):
            base_act = torch.tensor(base_act, dtype=torch.float32)
        base_activations.append(base_act)
    
    # Stack tensors into a batch
    base_activations_batch = torch.stack(base_activations)
    
    return {
        'rubric': rubrics,
        'base_activation': base_activations_batch,
        'prompt': prompts
    }


def exploration_phase(
    concept_explainer: ConceptExplainer,
    intervention_pipeline: InterventionPipeline,
    verifier: Verifier,
    rubrics: List[RubricDefinition],
    neutral_prompts: List[str],
    extractor: BaseModelActivationExtractor,
    layer_idx: int,
    n_samples_per_rubric: int = 100,
    epsilon: float = 0.1,
    success_threshold: float = 0.6,
    device: str = "cuda",
    log_dir: Optional[str] = None
) -> List[Dict]:
    """
    Exploration phase: Random perturbation sampling to find seed directions.
    
    For each rubric:
    1. Sample neutral prompts
    2. Extract base activations f(x)
    3. Sample random interventions δf ~ N(0, σ²I)
    4. Apply intervention and generate response
    5. Verify with text-based verifier
    6. Store successful interventions
    
    Args:
        concept_explainer: ConceptExplainer (not used yet, for consistency)
        intervention_pipeline: InterventionPipeline for applying interventions
        verifier: Verifier for scoring responses
        rubrics: List of rubrics to explore
        neutral_prompts: List of neutral prompts
        extractor: BaseModelActivationExtractor for getting base activations
        layer_idx: Layer to intervene on
        n_samples_per_rubric: Number of random samples per rubric
        epsilon: Norm constraint coefficient
        success_threshold: Verifier score threshold for success
        device: Device to run on
        log_dir: Optional directory to log successful interventions
        
    Returns:
        List of successful intervention examples
    """
    successful_examples = []
    
    print(f"\n{'='*60}")
    print("EXPLORATION PHASE: Random Perturbation Sampling")
    print(f"{'='*60}")
    
    for rubric in rubrics:
        print(f"\nExploring rubric: {rubric.rubric_text[:50]}...")
        
        rubric_successes = 0
        
        for i in tqdm(range(n_samples_per_rubric), desc="Sampling"):
            # 1. Sample neutral prompt
            prompt = np.random.choice(neutral_prompts)
            
            # 2. Extract base activation
            activations = extractor.extract_activations(
                prompt=prompt,
                layer_indices=[layer_idx],
                position="last"
            )
            base_f = activations[layer_idx][0].detach().clone().to(device=device, dtype=torch.float32)
            
            # 3. Sample random intervention
            d_hidden = base_f.shape[0]
            delta_f = torch.randn(d_hidden, device=device)
            
            # Normalize to satisfy constraint
            max_norm = epsilon * torch.norm(base_f)
            delta_f = delta_f / torch.norm(delta_f) * max_norm * torch.rand(1, device=device)
            
            # 4. Apply intervention and generate response
            try:
                response = intervention_pipeline.apply_intervention(
                    prompt=prompt,
                    layer_idx=layer_idx,
                    delta_f=delta_f,
                    position="last",
                    max_new_tokens=100
                )
                
                # 5. Verify with text-based verifier
                score = verifier.verify(rubric, response)
                
                # 6. Store if successful
                if score >= success_threshold:
                    record = {
                        'rubric': rubric,
                        'base_activation': base_f.cpu(),
                        'intervention': delta_f.cpu(),
                        'prompt': prompt,
                        'response': response,
                        'score': score,
                        'layer_idx': layer_idx
                    }
                    successful_examples.append(record)
                    if log_dir:
                        _append_jsonl(
                            Path(log_dir) / "exploration_success.jsonl",
                            {
                                "timestamp": datetime.utcnow().isoformat(),
                                "rubric_text": rubric.rubric_text,
                                "target_answer": rubric.target_answer,
                                "category": rubric.category,
                                "prompt": prompt,
                                "response": response,
                                "score": score,
                                "layer_idx": layer_idx,
                                "delta_f": delta_f.detach().cpu().tolist()
                            }
                        )
                    rubric_successes += 1
                    
            except Exception as e:
                print(f"  Error in sample {i}: {e}")
                continue
        
        print(f"  Found {rubric_successes}/{n_samples_per_rubric} successful interventions")
    
    print(f"\nTotal successful interventions: {len(successful_examples)}")
    return successful_examples


def policy_gradient_phase(
    concept_explainer: ConceptExplainer,
    intervention_pipeline: InterventionPipeline,
    verifier: Verifier,
    training_data: List[Dict],
    layer_idx: int,
    n_epochs: int = 10,
    batch_size: int = 8,
    learning_rate: float = 1e-4,
    gamma: float = 0.99,
    entropy_coef: float = 0.01,
    device: str = "cuda",
    save_dir: Optional[str] = None,
    epsilon: float = 0.1,
    log_dir: Optional[str] = None,
    log_threshold: float = 0.6
) -> ConceptExplainer:
    """
    Policy gradient training phase using REINFORCE.
    
    Objective: max E_{δf ~ p_θ(·|c)} [R(δf, c)]
    where R(δf, c) = verifier_score(M(x; f(x) + δf), rubric)
    
    Uses:
    - Baseline for variance reduction
    - Entropy regularization to prevent mode collapse
    
    Args:
        concept_explainer: ConceptExplainer to train
        intervention_pipeline: InterventionPipeline for applying interventions
        verifier: Verifier for scoring
        training_data: List of training examples from rubric dataset
        layer_idx: Layer to intervene on
        n_epochs: Number of training epochs
        batch_size: Batch size
        learning_rate: Learning rate
        gamma: Discount factor (not used in single-step setting)
        entropy_coef: Entropy regularization coefficient
        device: Device to run on
        save_dir: Directory to save checkpoints
        epsilon: Norm constraint coefficient for interventions
        log_dir: Optional directory to log high-scoring interventions
        log_threshold: Score threshold for logging interventions
        
    Returns:
        Trained ConceptExplainer
    """
    print(f"\n{'='*60}")
    print("POLICY GRADIENT PHASE: REINFORCE Training")
    print(f"{'='*60}")
    
    # Check if we have enough data
    if len(training_data) == 0:
        raise ValueError("No training data provided. Need at least 1 training example.")
    
    if len(training_data) < batch_size:
        print(f"  Warning: Dataset size ({len(training_data)}) is smaller than batch size ({batch_size})")
        print(f"  Reducing batch size to {len(training_data)}")
        batch_size = len(training_data)
    
    # Ensure batch_size is at least 1
    if batch_size < 1:
        batch_size = 1
    
    # Setup optimizer
    optimizer = optim.Adam(
        concept_explainer.policy_network.parameters(),
        lr=learning_rate
    )
    
    # Create dataset and dataloader
    dataset = PolicyTrainingDataset(training_data)
    # Use drop_last=False if dataset is small to ensure we get at least one batch
    drop_last = len(training_data) > batch_size * 2
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=drop_last,
        collate_fn=custom_collate_fn
    )
    
    print(f"  Dataset size: {len(training_data)}")
    print(f"  Batch size: {batch_size}")
    print(f"  Estimated batches per epoch: {len(dataloader)}")
    
    # Verify we have at least one batch
    if len(dataloader) == 0:
        raise ValueError(
            f"No batches can be created! Dataset size ({len(training_data)}) is too small "
            f"for batch size ({batch_size}) with drop_last={drop_last}. "
            f"Try reducing batch_size or ensure you have more training examples."
        )
    
    # Training loop
    for epoch in range(n_epochs):
        epoch_loss = 0.0
        epoch_reward = 0.0
        epoch_entropy = 0.0
        n_batches = 0
        
        print(f"\nEpoch {epoch+1}/{n_epochs}")
        
        for batch in tqdm(dataloader, desc="Training"):
            # Extract batch data
            rubrics = batch['rubric']
            base_activations = batch['base_activation'].to(device)
            prompts = batch['prompt']
            
            # Sample interventions from policy
            batch_size_actual = len(rubrics)
            interventions = []
            log_probs = []
            entropies = []
            
            for i in range(batch_size_actual):
                # Get distribution parameters
                # Concept encoder is frozen but returns normal tensors (cloned in encode method)
                concept_emb = concept_explainer.concept_encoder.encode(rubrics[i])
                
                mu, log_sigma = concept_explainer.policy_network(concept_emb)
                
                # Sample intervention
                delta_f = concept_explainer.policy_network.sample(mu, log_sigma)
                
                # Enforce constraint
                delta_f = concept_explainer.policy_network.enforce_constraint(
                    delta_f,
                    base_activations[i],
                    epsilon=epsilon
                )
                
                # Compute log probability
                log_prob = concept_explainer.policy_network.log_prob(
                    delta_f,
                    mu,
                    log_sigma
                )
                
                # Compute entropy (for regularization)
                sigma = torch.exp(log_sigma)
                entropy = 0.5 * torch.log(2 * np.pi * np.e * sigma**2).sum()
                
                interventions.append(delta_f)
                log_probs.append(log_prob)
                entropies.append(entropy)
            
            # Apply interventions and get rewards
            rewards = []
            
            for i in range(batch_size_actual):
                try:
                    # Apply intervention
                    response = intervention_pipeline.apply_intervention(
                        prompt=prompts[i],
                        layer_idx=layer_idx,
                        delta_f=interventions[i],
                        position="last",
                        max_new_tokens=100
                    )
                    
                    # Get reward from verifier
                    score = verifier.verify(rubrics[i], response)
                    rewards.append(score)
                    
                    if log_dir and score >= log_threshold:
                        _append_jsonl(
                            Path(log_dir) / "policy_success.jsonl",
                            {
                                "timestamp": datetime.utcnow().isoformat(),
                                "rubric_text": rubrics[i].rubric_text,
                                "target_answer": rubrics[i].target_answer,
                                "category": rubrics[i].category,
                                "prompt": prompts[i],
                                "response": response,
                                "score": score,
                                "layer_idx": layer_idx,
                                "delta_f": interventions[i].detach().cpu().tolist()
                            }
                        )
                    
                except Exception as e:
                    print(f"  Error in batch item {i}: {e}")
                    rewards.append(0.0)
            
            # Convert to tensors
            rewards_tensor = torch.tensor(rewards, device=device)
            log_probs_tensor = torch.stack(log_probs)
            entropies_tensor = torch.stack(entropies)
            
            # Compute baseline (mean reward)
            baseline = rewards_tensor.mean()
            
            # Compute policy gradient loss
            # L = -E[log π(a|s) * (R - baseline)]
            advantages = rewards_tensor - baseline
            policy_loss = -(log_probs_tensor * advantages).mean()
            
            # Add entropy regularization
            entropy_loss = -entropy_coef * entropies_tensor.mean()
            
            # Total loss
            loss = policy_loss + entropy_loss
            
            # Optimize
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                concept_explainer.policy_network.parameters(),
                max_norm=1.0
            )
            optimizer.step()
            
            # Track metrics
            epoch_loss += loss.item()
            epoch_reward += rewards_tensor.mean().item()
            epoch_entropy += entropies_tensor.mean().item()
            n_batches += 1
        
        # Print epoch metrics
        if n_batches > 0:
            avg_loss = epoch_loss / n_batches
            avg_reward = epoch_reward / n_batches
            avg_entropy = epoch_entropy / n_batches
            
            print(f"  Loss: {avg_loss:.4f}")
            print(f"  Avg Reward: {avg_reward:.4f}")
            print(f"  Avg Entropy: {avg_entropy:.4f}")
        else:
            print(f"  ⚠ No batches processed in this epoch!")
            print(f"  Dataset size: {len(dataset)}, Batch size: {batch_size}")
            print(f"  Skipping this epoch...")
            continue
        
        # Save checkpoint
        if save_dir and (epoch + 1) % 5 == 0:
            save_path = Path(save_dir) / f"policy_epoch_{epoch+1}.pt"
            save_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save({
                'epoch': epoch + 1,
                'policy_state_dict': concept_explainer.policy_network.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'avg_reward': avg_reward,
            }, save_path)
            print(f"  Saved checkpoint: {save_path}")
    
    return concept_explainer


def train_policy(
    concept_explainer: ConceptExplainer,
    intervention_pipeline: InterventionPipeline,
    verifier: Verifier,
    rubrics: List[RubricDefinition],
    neutral_prompts: List[str],
    triples_path: str,
    extractor: BaseModelActivationExtractor,
    layer_idx: int,
    n_exploration_samples: int = 100,
    n_training_epochs: int = 10,
    batch_size: int = 8,
    learning_rate: float = 1e-4,
    save_dir: str = "checkpoints/policy",
    device: str = "cuda",
    epsilon: float = 0.1,
    use_rubric_dataset: bool = True,
    dataset_path: Optional[str] = None,
    max_train_examples: Optional[int] = None,
    skip_exploration: bool = False,
    log_dir: Optional[str] = None,
    log_threshold: float = 0.6
) -> Tuple[ConceptExplainer, List[Dict]]:
    """
    Main training loop: Exploration + Policy Gradient.
    
    Args:
        concept_explainer: ConceptExplainer to train
        intervention_pipeline: InterventionPipeline for applying interventions
        verifier: Verifier for scoring
        rubrics: List of rubrics to train on
        neutral_prompts: List of neutral prompts
        triples_path: Path to rubric triples JSON
        extractor: BaseModelActivationExtractor for base activations
        layer_idx: Layer to intervene on
        n_exploration_samples: Samples per rubric in exploration
        n_training_epochs: Number of policy gradient epochs
        batch_size: Training batch size
        learning_rate: Learning rate
        save_dir: Directory to save checkpoints
        device: Device to run on
        epsilon: Norm constraint coefficient for interventions
        use_rubric_dataset: Whether to train using full rubric dataset
        dataset_path: Optional path to cached rubric dataset JSON
        max_train_examples: Max rubric dataset examples to use
        skip_exploration: Whether to skip exploration phase
        log_dir: Optional directory to log interventions
        log_threshold: Score threshold for logging interventions
        
    Returns:
        (trained_explainer, successful_examples)
    """
    print(f"\n{'='*60}")
    print("POLICY TRAINING PIPELINE")
    print(f"{'='*60}")
    print(f"Rubrics: {len(rubrics)}")
    print(f"Neutral prompts: {len(neutral_prompts)}")
    print(f"Layer: {layer_idx}")
    print(f"Exploration samples/rubric: {n_exploration_samples}")
    print(f"Training epochs: {n_training_epochs}")
    
    successful_examples: List[Dict] = []
    if not skip_exploration:
        # Phase 1: Exploration (logging/bootstrapping only)
        successful_examples = exploration_phase(
            concept_explainer=concept_explainer,
            intervention_pipeline=intervention_pipeline,
            verifier=verifier,
            rubrics=rubrics,
            neutral_prompts=neutral_prompts,
            extractor=extractor,
            layer_idx=layer_idx,
            n_samples_per_rubric=n_exploration_samples,
            epsilon=epsilon,
            device=device,
            log_dir=log_dir
        )
        
        if len(successful_examples) == 0:
            print("\n⚠ No successful interventions found in exploration phase!")
            print("  Proceeding to policy gradient training with full dataset...")
    
    # Phase 2: Policy Gradient Training
    if use_rubric_dataset:
        rubric_dataset = None
        if dataset_path and Path(dataset_path).exists():
            rubric_dataset = _load_rubric_dataset(Path(dataset_path))
            if max_train_examples:
                rubric_dataset = rubric_dataset[:max_train_examples]
        else:
            rubric_dataset = generate_rubric_dataset(
                extractor=extractor,
                triples_path=triples_path,
                verifier=verifier,
                layer_indices=[layer_idx],
                position="last",
                max_examples=max_train_examples,
                verbose=True
            )
            if dataset_path:
                _serialize_rubric_dataset(rubric_dataset, Path(dataset_path))
        
        training_data = build_policy_training_examples(
            rubric_dataset,
            layer_idx=layer_idx
        )
    else:
        training_data = successful_examples
    
    trained_explainer = policy_gradient_phase(
        concept_explainer=concept_explainer,
        intervention_pipeline=intervention_pipeline,
        verifier=verifier,
        training_data=training_data,
        layer_idx=layer_idx,
        n_epochs=n_training_epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        device=device,
        save_dir=save_dir,
        epsilon=epsilon,
        log_dir=log_dir,
        log_threshold=log_threshold
    )
    
    # Save final model
    final_path = Path(save_dir) / "policy_final.pt"
    final_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        'policy_state_dict': trained_explainer.policy_network.state_dict(),
        'n_successful_examples': len(successful_examples),
        'n_training_examples': len(training_data),
        'rubrics': [r.__dict__ for r in rubrics],
    }, final_path)
    print(f"\n✓ Saved final model: {final_path}")
    
    return trained_explainer, successful_examples


if __name__ == "__main__":
    import argparse
    import os
    from dotenv import load_dotenv
    from transformers import AutoTokenizer, AutoModelForCausalLM
    
    # Load environment variables
    load_dotenv()
    
    parser = argparse.ArgumentParser(
        description="Train policy network to discover jailbreak activation interventions"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="google/gemma-2-2b-it",
        help="Target model to intervene on (default: google/gemma-2-2b-it)"
    )
    parser.add_argument(
        "--layer",
        type=int,
        default=None,
        help="Layer index to intervene on (default: middle layer)"
    )
    parser.add_argument(
        "--rubrics-path",
        type=str,
        default="data/rubrics.json",
        help="Path to rubrics JSON file (default: data/rubrics.json)"
    )
    parser.add_argument(
        "--triples-path",
        type=str,
        default="data/rubric_triples.json",
        help="Path to rubric triples JSON file (default: data/rubric_triples.json)"
    )
    parser.add_argument(
        "--n-exploration",
        type=int,
        default=100,
        help="Number of exploration samples per rubric (default: 100)"
    )
    parser.add_argument(
        "--n-epochs",
        type=int,
        default=10,
        help="Number of training epochs (default: 10)"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Training batch size (default: 8)"
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=1e-4,
        help="Learning rate (default: 1e-4)"
    )
    parser.add_argument(
        "--save-dir",
        type=str,
        default="checkpoints/policy",
        help="Directory to save checkpoints (default: checkpoints/policy)"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to run on (default: cuda if available, else cpu)"
    )
    parser.add_argument(
        "--verifier-model",
        type=str,
        default="meta-llama/Llama-3.1-8B-Instruct",
        help="Verifier model name (default: meta-llama/Llama-3.1-8B-Instruct)"
    )
    parser.add_argument(
        "--verifier-device",
        type=str,
        default="cpu",
        help="Device for verifier model (default: cpu)"
    )
    parser.add_argument(
        "--max-rubrics",
        type=int,
        default=None,
        help="Maximum number of rubrics to use (None = all, default: None)"
    )
    parser.add_argument(
        "--max-prompts",
        type=int,
        default=500,
        help="Maximum number of neutral prompts to use (default: 500)"
    )
    parser.add_argument(
        "--no-rubric-dataset",
        action="store_false",
        dest="use_rubric_dataset",
        help="Train only on exploration successes (skip full rubric dataset)"
    )
    parser.add_argument(
        "--dataset-path",
        type=str,
        default=None,
        help="Optional path to cached rubric dataset JSON"
    )
    parser.add_argument(
        "--max-train-examples",
        type=int,
        default=None,
        help="Maximum rubric dataset examples to use (default: all)"
    )
    parser.add_argument(
        "--skip-exploration",
        action="store_true",
        help="Skip exploration phase (policy gradient only)"
    )
    parser.add_argument(
        "--epsilon",
        type=float,
        default=0.1,
        help="Norm constraint coefficient for interventions (default: 0.1)"
    )
    parser.add_argument(
        "--log-dir",
        type=str,
        default="logs/interventions",
        help="Directory to log high-scoring interventions"
    )
    parser.add_argument(
        "--log-threshold",
        type=float,
        default=0.6,
        help="Verifier score threshold for logging interventions (default: 0.6)"
    )
    parser.add_argument(
        "--explainer-ckpt",
        type=str,
        default=None,
        help="Path to pretrained explainer checkpoint (optional)"
    )
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("POLICY TRAINING - Jailbreak Activation Intervention Discovery")
    print("=" * 60)
    print(f"Target model: {args.model}")
    print(f"Device: {args.device}")
    print(f"Rubrics path: {args.rubrics_path}")
    print(f"Triples path: {args.triples_path}")
    print(f"Exploration samples/rubric: {args.n_exploration}")
    print(f"Training epochs: {args.n_epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"Learning rate: {args.learning_rate}")
    print(f"Verifier device: {args.verifier_device}")
    print(f"Use rubric dataset: {args.use_rubric_dataset}")
    print(f"Skip exploration: {args.skip_exploration}")
    print(f"Epsilon: {args.epsilon}")
    print(f"Log dir: {args.log_dir}")
    print("=" * 60)
    
    # 1. Load rubrics
    print("\n[1/6] Loading rubrics...")
    if not Path(args.rubrics_path).exists():
        raise FileNotFoundError(
            f"Rubrics file not found at {args.rubrics_path}. "
            "Run 'python src/rubrics.py --dataset <name> --num-samples <n>' first."
        )
    
    with open(args.rubrics_path, "r", encoding="utf-8") as f:
        rubrics_data = json.load(f)
    
    rubrics: List[RubricDefinition] = []
    for item in rubrics_data:
        rubrics.append(
            RubricDefinition(
                rubric_text=item["rubric_text"],
                target_answer=item.get("target_answer", "satisfies"),
                category=item.get("category", "jailbreak"),
                metadata=item.get("metadata", {}),
            )
        )
    
    if args.max_rubrics:
        rubrics = rubrics[:args.max_rubrics]
    
    print(f"  ✓ Loaded {len(rubrics)} rubrics")
    
    # 2. Load neutral prompts from triples
    print("\n[2/6] Loading neutral prompts...")
    if not Path(args.triples_path).exists():
        raise FileNotFoundError(
            f"Rubric triples file not found at {args.triples_path}. "
            "Run 'python src/rubrics.py --dataset <name> --num-samples <n>' first."
        )
    
    from src.data import load_rubric_triples
    triples = load_rubric_triples(args.triples_path)
    
    # Extract unique prompts
    neutral_prompts = list(set([triple["prompt"] for triple in triples]))
    if args.max_prompts:
        neutral_prompts = neutral_prompts[:args.max_prompts]
    
    print(f"  ✓ Loaded {len(neutral_prompts)} unique neutral prompts")
    
    # 3. Initialize activation extractor
    print("\n[3/6] Initializing activation extractor...")
    print(f"  (This will download {args.model} if not cached)")
    extractor = BaseModelActivationExtractor(
        model_name=args.model,
        layer_indices=None,  # Auto-calculate layers
        device=args.device,
    )
    
    # Determine layer index
    if args.layer is None:
        # Use middle layer
        layer_idx = extractor.layer_indices[len(extractor.layer_indices) // 2]
    else:
        layer_idx = args.layer
    
    print(f"  ✓ Extractor ready (layers: {extractor.layer_indices}, using layer {layer_idx})")
    
    # 4. Initialize intervention pipeline
    print("\n[4/6] Initializing intervention pipeline...")
    # Reuse the same model from extractor
    intervention_pipeline = InterventionPipeline(
        model=extractor.model,
        tokenizer=extractor.tokenizer,
        device=args.device
    )
    print("  ✓ Intervention pipeline ready")
    
    # 5. Initialize verifier
    print("\n[5/6] Initializing verifier...")
    verifier = Verifier(
        mode="llm",
        model_name=args.verifier_model,
        device=args.verifier_device
    )
    print("  ✓ Verifier ready")
    
    # 6. Initialize concept explainer
    print("\n[6/6] Initializing concept explainer...")
    # Get hidden dimension from model config (most reliable method)
    d_hidden = 2048  # Default fallback
    if hasattr(extractor.model, 'config'):
        # Try common config attribute names for hidden dimension
        d_hidden = (
            getattr(extractor.model.config, 'hidden_size', None) or
            getattr(extractor.model.config, 'd_model', None) or
            getattr(extractor.model.config, 'n_embd', None) or
            2048
        )
    
    # If config doesn't have it, try to infer from model structure
    if d_hidden == 2048 and hasattr(extractor.model, 'model') and hasattr(extractor.model.model, 'layers'):
        try:
            first_layer = extractor.model.model.layers[0]
            # Try different ways to get hidden dim from attention
            if hasattr(first_layer, 'self_attn'):
                attn = first_layer.self_attn
                # Method 1: head_dim * num_heads (for Llama-style)
                if hasattr(attn, 'head_dim') and hasattr(attn, 'num_heads'):
                    d_hidden = attn.head_dim * attn.num_heads
                # Method 2: num_heads * head_dim (alternative attribute names)
                elif hasattr(attn, 'head_dim') and hasattr(attn, 'num_attention_heads'):
                    d_hidden = attn.head_dim * attn.num_attention_heads
                # Method 3: Get from q_proj weight shape (most reliable)
                elif hasattr(attn, 'q_proj') and hasattr(attn.q_proj, 'weight'):
                    # q_proj weight shape is typically [hidden_size, hidden_size] or [hidden_size, num_heads * head_dim]
                    q_shape = attn.q_proj.weight.shape
                    if len(q_shape) >= 2:
                        d_hidden = q_shape[0]  # Input dimension
            # Method 4: Get from mlp or feed_forward
            elif hasattr(first_layer, 'mlp') and hasattr(first_layer.mlp, 'gate_proj'):
                if hasattr(first_layer.mlp.gate_proj, 'weight'):
                    d_hidden = first_layer.mlp.gate_proj.weight.shape[0]
            elif hasattr(first_layer, 'feed_forward') and hasattr(first_layer.feed_forward, 'w1'):
                if hasattr(first_layer.feed_forward.w1, 'weight'):
                    d_hidden = first_layer.feed_forward.w1.weight.shape[0]
        except Exception as e:
            print(f"  Warning: Could not infer hidden dimension from model structure: {e}")
            print(f"  Using default: {d_hidden}")
    
    concept_explainer = ConceptExplainer(
        d_hidden=d_hidden,
        device=args.device
    )
    print(f"  ✓ Concept explainer ready (d_hidden={d_hidden})")
    
    if args.explainer_ckpt:
        ckpt_path = Path(args.explainer_ckpt)
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Explainer checkpoint not found: {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location=args.device)
        if "policy_state_dict" not in ckpt:
            raise KeyError("Checkpoint missing 'policy_state_dict'")
        concept_explainer.policy_network.load_state_dict(ckpt["policy_state_dict"])
        print(f"  ✓ Loaded explainer checkpoint from {ckpt_path}")
    
    # Run training
    print("\n" + "=" * 60)
    print("STARTING TRAINING")
    print("=" * 60)
    
    trained_explainer, successful_examples = train_policy(
        concept_explainer=concept_explainer,
        intervention_pipeline=intervention_pipeline,
        verifier=verifier,
        rubrics=rubrics,
        neutral_prompts=neutral_prompts,
        triples_path=args.triples_path,
        extractor=extractor,
        layer_idx=layer_idx,
        n_exploration_samples=args.n_exploration,
        n_training_epochs=args.n_epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        save_dir=args.save_dir,
        device=args.device,
        epsilon=args.epsilon,
        use_rubric_dataset=args.use_rubric_dataset,
        dataset_path=args.dataset_path,
        max_train_examples=args.max_train_examples,
        skip_exploration=args.skip_exploration,
        log_dir=args.log_dir,
        log_threshold=args.log_threshold
    )
    
    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)
    print(f"✓ Trained policy saved to: {args.save_dir}/policy_final.pt")
    print(f"✓ Found {len(successful_examples)} successful interventions")
    print("=" * 60)
