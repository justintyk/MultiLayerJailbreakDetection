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

from typing import List, Dict, Tuple, Optional
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
from src.data import BaseModelActivationExtractor


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
    success_threshold: float = 0.7,
    device: str = "cuda"
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
            base_f = torch.tensor(
                activations[layer_idx][0],
                dtype=torch.float32,
                device=device
            )
            
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
                    successful_examples.append({
                        'rubric': rubric,
                        'base_activation': base_f.cpu(),
                        'intervention': delta_f.cpu(),
                        'prompt': prompt,
                        'response': response,
                        'score': score,
                        'layer_idx': layer_idx
                    })
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
    save_dir: Optional[str] = None
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
        training_data: List of training examples from exploration
        layer_idx: Layer to intervene on
        n_epochs: Number of training epochs
        batch_size: Batch size
        learning_rate: Learning rate
        gamma: Discount factor (not used in single-step setting)
        entropy_coef: Entropy regularization coefficient
        device: Device to run on
        save_dir: Directory to save checkpoints
        
    Returns:
        Trained ConceptExplainer
    """
    print(f"\n{'='*60}")
    print("POLICY GRADIENT PHASE: REINFORCE Training")
    print(f"{'='*60}")
    
    # Setup optimizer
    optimizer = optim.Adam(
        concept_explainer.policy_network.parameters(),
        lr=learning_rate
    )
    
    # Create dataset and dataloader
    dataset = PolicyTrainingDataset(training_data)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True
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
                with torch.no_grad():
                    concept_emb = concept_explainer.concept_encoder.encode(rubrics[i])
                
                mu, log_sigma = concept_explainer.policy_network(concept_emb)
                
                # Sample intervention
                delta_f = concept_explainer.policy_network.sample(mu, log_sigma)
                
                # Enforce constraint
                delta_f = concept_explainer.policy_network.enforce_constraint(
                    delta_f,
                    base_activations[i],
                    epsilon=0.1
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
        avg_loss = epoch_loss / n_batches
        avg_reward = epoch_reward / n_batches
        avg_entropy = epoch_entropy / n_batches
        
        print(f"  Loss: {avg_loss:.4f}")
        print(f"  Avg Reward: {avg_reward:.4f}")
        print(f"  Avg Entropy: {avg_entropy:.4f}")
        
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
    extractor: BaseModelActivationExtractor,
    layer_idx: int,
    n_exploration_samples: int = 100,
    n_training_epochs: int = 10,
    batch_size: int = 8,
    learning_rate: float = 1e-4,
    save_dir: str = "checkpoints/policy",
    device: str = "cuda"
) -> Tuple[ConceptExplainer, List[Dict]]:
    """
    Main training loop: Exploration + Policy Gradient.
    
    Args:
        concept_explainer: ConceptExplainer to train
        intervention_pipeline: InterventionPipeline for applying interventions
        verifier: Verifier for scoring
        rubrics: List of rubrics to train on
        neutral_prompts: List of neutral prompts
        extractor: BaseModelActivationExtractor for base activations
        layer_idx: Layer to intervene on
        n_exploration_samples: Samples per rubric in exploration
        n_training_epochs: Number of policy gradient epochs
        batch_size: Training batch size
        learning_rate: Learning rate
        save_dir: Directory to save checkpoints
        device: Device to run on
        
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
    
    # Phase 1: Exploration
    successful_examples = exploration_phase(
        concept_explainer=concept_explainer,
        intervention_pipeline=intervention_pipeline,
        verifier=verifier,
        rubrics=rubrics,
        neutral_prompts=neutral_prompts,
        extractor=extractor,
        layer_idx=layer_idx,
        n_samples_per_rubric=n_exploration_samples,
        device=device
    )
    
    if len(successful_examples) == 0:
        print("\n⚠ No successful interventions found in exploration phase!")
        print("  Try adjusting success_threshold or increasing n_samples")
        return concept_explainer, []
    
    # Phase 2: Policy Gradient Training
    trained_explainer = policy_gradient_phase(
        concept_explainer=concept_explainer,
        intervention_pipeline=intervention_pipeline,
        verifier=verifier,
        training_data=successful_examples,
        layer_idx=layer_idx,
        n_epochs=n_training_epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        device=device,
        save_dir=save_dir
    )
    
    # Save final model
    final_path = Path(save_dir) / "policy_final.pt"
    final_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        'policy_state_dict': trained_explainer.policy_network.state_dict(),
        'n_successful_examples': len(successful_examples),
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
        layer_indices=None  # Auto-calculate layers
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
        device=args.device
    )
    print("  ✓ Verifier ready")
    
    # 6. Initialize concept explainer
    print("\n[6/6] Initializing concept explainer...")
    # Get hidden dimension from model
    if hasattr(extractor.model, 'model') and hasattr(extractor.model.model, 'layers'):
        # Get hidden dim from first layer
        d_hidden = extractor.model.model.layers[0].self_attn.head_dim * extractor.model.model.layers[0].self_attn.num_heads
    else:
        # Fallback: try to get from config
        if hasattr(extractor.model, 'config'):
            d_hidden = getattr(extractor.model.config, 'hidden_size', 2048)
        else:
            d_hidden = 2048  # Default fallback
    
    concept_explainer = ConceptExplainer(
        d_hidden=d_hidden,
        device=args.device
    )
    print(f"  ✓ Concept explainer ready (d_hidden={d_hidden})")
    
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
        extractor=extractor,
        layer_idx=layer_idx,
        n_exploration_samples=args.n_exploration,
        n_training_epochs=args.n_epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        save_dir=args.save_dir,
        device=args.device
    )
    
    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)
    print(f"✓ Trained policy saved to: {args.save_dir}/policy_final.pt")
    print(f"✓ Found {len(successful_examples)} successful interventions")
    print("=" * 60)
