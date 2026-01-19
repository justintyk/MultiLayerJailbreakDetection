"""
Module 3: Training - Fine-tune Activation Oracle on multi-layer jailbreak dataset.

This module wires together:
    - the multi-layer activation dataset from src.data
    - the Activation Oracle wrapper from src.models
    - supervised fine-tuning that optimizes a jailbreak detection objective

This implementation uses TRL's SFTTrainer with a custom data collator for
full activation-conditioned training following the LatentQA paradigm.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Any
from dataclasses import dataclass

import torch
from torch.utils.data import Dataset

from transformers import TrainingArguments
from trl import SFTTrainer
from data import load_dataset, generate_activation_multilayer_dataset, BaseModelActivationExtractor
from models import MultiLayerActivationOracle


# Extended label taxonomy for richer jailbreak classification
LABEL_MAP: Dict[str, int] = {
    "SAFE": 0,
    "JAILBREAK": 1,
    "JAILBREAK_ROLEPLAY": 2,
    "JAILBREAK_OBFUSCATION": 3,
    "JAILBREAK_PROMPT_INJECTION": 4,
}

LABEL_MAP_BINARY: Dict[str, int] = {
    "SAFE": 0,
    "JAILBREAK": 1,
}


class MultiLayerJailbreakDataset(Dataset):
    """
    Dataset wrapper for activation examples with full activation data.

    Each item contains:
        - activations_by_layer: dict[int -> tensor]
        - label: str (e.g., "SAFE", "JAILBREAK")
        - text: str (prompt + completion for training)
    """

    def __init__(
        self, 
        examples: List[dict],
        oracle: MultiLayerActivationOracle,
        use_extended_taxonomy: bool = False,
    ) -> None:
        self.examples = examples
        self.oracle = oracle
        self.use_extended_taxonomy = use_extended_taxonomy
        self.label_map = LABEL_MAP if use_extended_taxonomy else LABEL_MAP_BINARY

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int):
        ex = self.examples[idx]
        
        # Convert activations to tensors
        activations_by_layer = {
            int(layer): torch.tensor(vectors, dtype=torch.float32)
            for layer, vectors in ex["activations_by_layer"].items()
        }
        
        label_str = ex["label"]
        
        # Build completion text based on label
        if self.use_extended_taxonomy:
            completions = {
                "SAFE": "SAFE - No jailbreak detected.",
                "JAILBREAK": "JAILBREAK - Generic jailbreak attempt detected.",
                "JAILBREAK_ROLEPLAY": "JAILBREAK_ROLEPLAY - Roleplay-based safety bypass detected.",
                "JAILBREAK_OBFUSCATION": "JAILBREAK_OBFUSCATION - Obfuscation-based jailbreak detected.",
                "JAILBREAK_PROMPT_INJECTION": "JAILBREAK_PROMPT_INJECTION - Prompt injection attack detected.",
            }
            completion = completions.get(label_str, completions["JAILBREAK"])
        else:
            if label_str == "SAFE":
                completion = "SAFE - This is a benign interaction."
            else:
                completion = "JAILBREAK - This is a jailbreak attempt."
        
        # Build oracle prompt with placeholders
        num_layers = len(activations_by_layer)
        layer_desc = "+".join([f"L{layer}" for layer in sorted(activations_by_layer.keys())])
        
        question = "Analyze this activation pattern. Is it a jailbreak?"
        prompt = self.oracle.build_oracle_prompt(
            layer_description=layer_desc,
            num_placeholders=num_layers,
            question=question,
        )
        
        # Combine for training
        text = f"{prompt}\n{completion}"
        
        return {
            "text": text,
            "activations_by_layer": activations_by_layer,
            "label": label_str,
        }


class ActivationInjectionCollator:
    """
    Custom data collator that injects activation vectors during training.
    
    This is the key component that makes activation-conditioned training work.
    It replaces placeholder tokens with actual activation embeddings during
    the forward pass, following the LatentQA/Activation Oracle paradigm.
    """
    
    def __init__(
        self,
        tokenizer,
        oracle: MultiLayerActivationOracle,
        max_length: int = 512,
    ):
        self.tokenizer = tokenizer
        self.oracle = oracle
        self.max_length = max_length
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Get placeholder token ID
        self.placeholder_token_id = self.tokenizer.encode(
            self.oracle.placeholder_token,
            add_special_tokens=False
        )[0]
    
    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        """
        Process a batch and inject activations into embeddings.
        
        Args:
            batch: List of examples from MultiLayerJailbreakDataset
            
        Returns:
            Dict with inputs_embeds, attention_mask, and labels
        """
        # Tokenize all texts in batch
        texts = [ex["text"] for ex in batch]
        tokenized = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        
        input_ids = tokenized["input_ids"].to(self.device)
        attention_mask = tokenized["attention_mask"].to(self.device)
        
        # Get embedding layer
        if hasattr(self.oracle.model, 'get_input_embeddings'):
            embed_layer = self.oracle.model.get_input_embeddings()
        elif hasattr(self.oracle.model.base_model, 'get_input_embeddings'):
            embed_layer = self.oracle.model.base_model.get_input_embeddings()
        else:
            raise AttributeError("Cannot find embedding layer")
        
        # Get base embeddings
        embeddings = embed_layer(input_ids)  # [batch_size, seq_len, hidden_dim]
        
        # Inject activations for each example in batch
        for i, ex in enumerate(batch):
            activations_by_layer = ex["activations_by_layer"]
            
            # Aggregate multi-layer activations
            agg_acts = self.oracle._encode_multi_layer_activations(
                activations_by_layer
            )  # [K, D] where K = num layers
            
            # Find placeholder positions in this example
            placeholder_positions = (input_ids[i] == self.placeholder_token_id).nonzero(as_tuple=True)[0]
            
            # Inject activation vectors at placeholder positions
            num_to_inject = min(len(placeholder_positions), agg_acts.shape[0])
            for j in range(num_to_inject):
                pos = placeholder_positions[j]
                embeddings[i, pos, :] = agg_acts[j, :]
        
        # Create labels (same as input_ids for causal LM training)
        labels = input_ids.clone()
        
        return {
            "inputs_embeds": embeddings,
            "attention_mask": attention_mask,
            "labels": labels,
        }


def train_full_oracle_sft(
    n_examples: int = 1000,
    batch_size: int = 8,
    num_epochs: int = 3,
    learning_rate: float = 2e-5,
    use_extended_taxonomy: bool = False,
    output_dir: str = "./oracle_sft_output",
    warmup_ratio: float = 0.1,
):
    """
    Full end-to-end supervised fine-tuning of the Activation Oracle model.
    
    This trains the oracle to generate natural language outputs about jailbreak
    detection, following the LatentQA paradigm. The oracle learns to condition
    on multi-layer activations and produce interpretable safety judgments.
    
    Args:
        n_examples: Total number of examples to generate
        batch_size: Training batch size (smaller for full model training)
        num_epochs: Number of training epochs
        learning_rate: Learning rate (lower for full model fine-tuning)
        use_extended_taxonomy: Whether to use extended jailbreak categories
        output_dir: Directory to save model checkpoints
        warmup_ratio: Ratio of total steps to use for warmup
    """
    print("\n" + "="*60)
    print("Full Activation Oracle SFT Training")
    print("="*60)
    
    # 1. Generate dataset with extracted activations
    print("\nExtracting activations from base model...")
    extractor = BaseModelActivationExtractor(
        model_name="google/gemma-2-2b-it",
        layer_indices=[7, 12, 23]
    )
    
    examples = generate_activation_multilayer_dataset(
        extractor=extractor,
        n_jailbreak=n_examples // 2,
        n_safe=n_examples // 2,
    )
    print(f"Generated {len(examples)} examples with extracted activations")
    
    # 2. Initialize Activation Oracle
    print("\nLoading Activation Oracle model...")
    oracle = MultiLayerActivationOracle()
    
    # 3. Create datasets with activation data
    from sklearn.model_selection import train_test_split
    train_examples, val_examples = train_test_split(examples, test_size=0.2, random_state=42)
    
    train_dataset = MultiLayerJailbreakDataset(
        train_examples,
        oracle,
        use_extended_taxonomy=use_extended_taxonomy,
    )
    val_dataset = MultiLayerJailbreakDataset(
        val_examples,
        oracle,
        use_extended_taxonomy=use_extended_taxonomy,
    )
    
    print(f"Train examples: {len(train_examples)}, Val examples: {len(val_examples)}")
    
    # 4. Create custom data collator for activation injection
    print("\nInitializing custom data collator with activation injection...")
    data_collator = ActivationInjectionCollator(
        tokenizer=oracle.tokenizer,
        oracle=oracle,
        max_length=512,
    )
    
    # 5. Setup training arguments with learning rate scheduling
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        learning_rate=learning_rate,
        warmup_ratio=warmup_ratio,
        lr_scheduler_type="cosine",
        logging_steps=10,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        fp16=True,
        gradient_accumulation_steps=2,
        report_to="none",  # Set to "wandb" if you want W&B logging
    )
    
    # 6. Create SFTTrainer with custom data collator
    print("\nInitializing SFTTrainer with activation-conditioned training...")
    trainer = SFTTrainer(
        model=oracle.model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
        dataset_text_field="text",
        max_seq_length=512,
    )
    
    # 7. Train
    print("\nStarting training...")
    print("✓ Using full activation injection during training")
    trainer.train()
    
    # 8. Save final model
    print(f"\nSaving final model to {output_dir}/final_model")
    trainer.save_model(f"{output_dir}/final_model")
    
    print("\nFull Oracle SFT training complete.")
    return trainer


def main(
    n_examples: int = 1000,
    use_extended_taxonomy: bool = False,
):
    """
    Main training entry point.
    
    Args:
        n_examples: Number of examples to generate
        use_extended_taxonomy: Whether to use extended jailbreak categories
    """
    print("\n" + "="*60)
    print("Module 3: Multi-Layer Jailbreak Detection Training")
    print("="*60)
    print(f"\nExamples: {n_examples}")
    print(f"Taxonomy: {'Extended' if use_extended_taxonomy else 'Binary'}")
    print()
    
    print("Training full Activation Oracle with SFT...")
    trainer = train_full_oracle_sft(
        n_examples=n_examples,
        use_extended_taxonomy=use_extended_taxonomy,
    )
    print("\n✓ Full Oracle SFT training complete")
    print("  Model saved to: oracle_sft_output/final_model")
    
    print("\n" + "="*60)
    print("Module 3: Training Complete")
    print("="*60)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Train multi-layer jailbreak detector")
    parser.add_argument(
        "--n_examples",
        type=int,
        default=1000,
        help="Number of examples to generate"
    )
    parser.add_argument(
        "--extended_taxonomy",
        action="store_true",
        help="Use extended jailbreak taxonomy instead of binary classification"
    )
    
    args = parser.parse_args()
    
    main(
        n_examples=args.n_examples,
        use_extended_taxonomy=args.extended_taxonomy,
    )
