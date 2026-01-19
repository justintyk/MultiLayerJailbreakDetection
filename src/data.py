"""
Module 1: Data & Tasks - Generate multi-layer jailbreak dataset.

This module defines the data schema and generators for our multi-layer 
activation-level jailbreak detection task.

The key idea is that each example contains:
    - activations_by_layer: dict[layer_idx -> tensor-like list of floats]
    - label:                "JAILBREAK" or "SAFE"
    - meta:                 optional metadata (prompt, source, model, etc.)
    - prompt:               the actual text prompt that generated activations

This module supports activation extraction from base LLMs via forward hooks.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Dict, List, Literal, TypedDict, Optional, Any
import random
import json
import pickle
from pathlib import Path

import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForCausalLM

Label = Literal["JAILBREAK", "SAFE"]


class ExampleDict(TypedDict):
    activations_by_layer: Dict[int, List[List[float]]]
    label: Label
    meta: Dict[str, str]
    prompt: str
    prompt_source: str
    model_name: str


@dataclass
class MultiLayerExample:
    """
    Structured representation of one dataset example.

    - activations_by_layer: maps each layer index (e.g., 7, 12, 23) to a
      list of activation vectors extracted from the model.
    - label: "JAILBREAK" or "SAFE".
    - prompt: the actual text prompt that generated these activations.
    - prompt_source: dataset origin (e.g., "jailbreakbench", "safe_generated").
    - model_name: which base LLM was used for activation extraction.
    - meta: optional additional metadata.
    """
    activations_by_layer: Dict[int, List[List[float]]]
    label: Label
    prompt: str
    prompt_source: str
    model_name: str
    meta: Dict[str, str]

    def to_dict(self) -> ExampleDict:
        return {
            "activations_by_layer": self.activations_by_layer,
            "label": self.label,
            "prompt": self.prompt,
            "prompt_source": self.prompt_source,
            "model_name": self.model_name,
            "meta": self.meta,
        }


class BaseModelActivationExtractor:
    """
    Extract activations from a base LLM at specified layers.
    
    This class loads a base model (e.g., Gemma, Llama) and registers forward
    hooks to capture hidden states at specified layers during inference.
    """
    
    def __init__(
        self,
        model_name: str = "google/gemma-2-2b-it",
        layer_indices: Optional[List[int]] = None,
    ):
        """
        Initialize the activation extractor.
        
        Args:
            model_name: HuggingFace model identifier for the base LLM.
            layer_indices: Which layers to extract activations from.
                          If None, defaults to [7, 12, 23].
        """
        self.model_name = model_name
        self.device = torch.device("cuda")
        self.layer_indices = layer_indices or [7, 12, 23]
        
        # Load tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
        )
        self.model.to(self.device)
        self.model.eval()
        
        # Storage for captured activations
        self.activations: Dict[int, torch.Tensor] = {}
        self.hooks = []
        
    def _register_hooks(self, layer_indices: List[int]) -> None:
        """Register forward hooks on specified layers to capture activations."""
        self._clear_hooks()
        self.activations = {}
        
        # Get the model's layers (works for most transformer architectures)
        if hasattr(self.model, 'model') and hasattr(self.model.model, 'layers'):
            layers = self.model.model.layers  # Gemma, Llama structure
        elif hasattr(self.model, 'transformer') and hasattr(self.model.transformer, 'h'):
            layers = self.model.transformer.h  # GPT-2 structure
        else:
            raise ValueError(f"Unsupported model architecture: {self.model_name}")
        
        def make_hook(layer_idx: int):
            def hook(module, input, output):
                # output is typically a tuple; first element is hidden states
                hidden_states = output[0] if isinstance(output, tuple) else output
                self.activations[layer_idx] = hidden_states.detach()
            return hook
        
        for layer_idx in layer_indices:
            if layer_idx >= len(layers):
                raise ValueError(
                    f"Layer index {layer_idx} out of range for model with {len(layers)} layers"
                )
            hook = layers[layer_idx].register_forward_hook(make_hook(layer_idx))
            self.hooks.append(hook)
    
    def _clear_hooks(self) -> None:
        """Remove all registered hooks."""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []
    
    def extract_activations(
        self,
        prompt: str,
        layer_indices: Optional[List[int]] = None,
        position: str = "last",
    ) -> Dict[int, torch.Tensor]:
        """
        Extract activations from the base model for a given prompt.
        
        Args:
            prompt: Text prompt to process.
            layer_indices: Which layers to extract from. If None, uses self.layer_indices.
            position: Which token positions to extract:
                     - "last": only the last token position
                     - "all": all token positions
                     - int: specific position index
        
        Returns:
            Dictionary mapping layer_idx -> tensor of shape [num_positions, hidden_dim]
        """
        if layer_indices is None:
            layer_indices = self.layer_indices
            
        self._register_hooks(layer_indices)
        
        # Tokenize and run forward pass
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            padding=True,
            truncation=True,
        ).to(self.device)
        
        with torch.no_grad():
            _ = self.model(**inputs)
        
        # Extract activations at specified positions
        result = {}
        for layer_idx, acts in self.activations.items():
            # acts shape: [batch_size, seq_len, hidden_dim]
            # We assume batch_size = 1
            acts = acts[0]  # [seq_len, hidden_dim]
            
            if position == "last":
                result[layer_idx] = acts[-1:, :]  # [1, hidden_dim]
            elif position == "all":
                result[layer_idx] = acts  # [seq_len, hidden_dim]
            elif isinstance(position, int):
                result[layer_idx] = acts[position:position+1, :]  # [1, hidden_dim]
            else:
                raise ValueError(f"Invalid position: {position}")
        
        self._clear_hooks()
        return result
    
    def __del__(self):
        """Cleanup hooks on deletion."""
        self._clear_hooks()


def load_jailbreak_prompts(
    dataset_name: str = "jailbreakbench",
    split: str = "all",
    max_examples: Optional[int] = None,
) -> List[Dict[str, str]]:
    """
    Load jailbreak prompts from established datasets.
    
    Args:
        dataset_name: Name of the dataset ("jailbreakbench" supported).
        split: Which split to load ("all", "harmful", etc.).
        max_examples: Maximum number of examples to load (None = all).
    
    Returns:
        List of dicts with keys: "prompt", "source", "category" (optional).
    """
    prompts = []
    
    if dataset_name == "jailbreakbench":
        try:
            import jailbreakbench as jbb
            
            # Load JailbreakBench behaviors
            behaviors = jbb.read_dataset()
            
            for behavior in behaviors:
                # Each behavior has a 'Behavior' field with the harmful request
                prompts.append({
                    "prompt": behavior.Behavior,
                    "source": "jailbreakbench",
                    "category": getattr(behavior, "Category", "unknown"),
                })
                
                if max_examples and len(prompts) >= max_examples:
                    break
                    
        except ImportError:
            raise ImportError(
                "jailbreakbench package not installed. "
                "Install with: pip install jailbreakbench"
            )
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")
    
    return prompts


def generate_safe_prompts(
    n_prompts: int = 100,
    include_hard_negatives: bool = True,
    dataset_sources: List[str] = ["alpaca", "dolly"],
) -> List[Dict[str, str]]:
    """
    Load safe prompts from instruction-following datasets (Alpaca, Dolly).
    
    Args:
        n_prompts: Number of safe prompts to load.
        include_hard_negatives: Whether to include prompts that are
                               structurally similar to jailbreaks but safe.
        dataset_sources: Which datasets to load from. Options: "alpaca", "dolly".
    
    Returns:
        List of dicts with keys: "prompt", "source", "category".
    """
    prompts = []
    
    # Load from Alpaca dataset (Stanford, 52k instruction-following examples)
    if "alpaca" in dataset_sources:
        try:
            from datasets import load_dataset
            
            # Load cleaned Alpaca dataset from HuggingFace
            alpaca_data = load_dataset("tatsu-lab/alpaca", split="train")
            
            # Sample from Alpaca
            n_alpaca = min(n_prompts // len(dataset_sources), len(alpaca_data))
            alpaca_sample = alpaca_data.shuffle(seed=42).select(range(n_alpaca))
            
            for example in alpaca_sample:
                # Alpaca format: instruction + optional input -> output
                instruction = example["instruction"]
                input_text = example.get("input", "")
                
                # Combine instruction and input if input exists
                if input_text:
                    prompt = f"{instruction}\n\nInput: {input_text}"
                else:
                    prompt = instruction
                
                prompts.append({
                    "prompt": prompt,
                    "source": "alpaca",
                    "category": "instruction_following",
                })
                
                if len(prompts) >= n_prompts:
                    break
                    
        except ImportError:
            print("Warning: 'datasets' package not installed. Skipping Alpaca dataset.")
            print("Install with: pip install datasets")
        except Exception as e:
            print(f"Warning: Could not load Alpaca dataset: {e}")
    
    # Load from Dolly dataset (Databricks, 15k human-generated instruction pairs)
    if "dolly" in dataset_sources and len(prompts) < n_prompts:
        try:
            from datasets import load_dataset
            
            # Load Dolly 15k dataset from HuggingFace
            dolly_data = load_dataset("databricks/databricks-dolly-15k", split="train")
            
            # Sample from Dolly
            n_dolly = min(n_prompts - len(prompts), len(dolly_data))
            dolly_sample = dolly_data.shuffle(seed=42).select(range(n_dolly))
            
            for example in dolly_sample:
                # Dolly format: instruction + context -> response
                instruction = example["instruction"]
                context = example.get("context", "")
                category = example.get("category", "unknown")
                
                # Combine instruction and context if context exists
                if context:
                    prompt = f"{instruction}\n\nContext: {context}"
                else:
                    prompt = instruction
                
                prompts.append({
                    "prompt": prompt,
                    "source": "dolly",
                    "category": category,
                })
                
                if len(prompts) >= n_prompts:
                    break
                    
        except ImportError:
            print("Warning: 'datasets' package not installed. Skipping Dolly dataset.")
            print("Install with: pip install datasets")
        except Exception as e:
            print(f"Warning: Could not load Dolly dataset: {e}")
    
    # Add hard negatives if requested and we still need more prompts
    if include_hard_negatives and len(prompts) < n_prompts:
        hard_negative_templates = [
            "Pretend you are a teacher and explain {topic}.",
            "Imagine you are writing a fictional story about {topic}.",
            "Act as a historian and describe {topic}.",
            "You are a helpful assistant. Please explain {topic}.",
            "In a hypothetical scenario where you're teaching a class, how would you explain {topic}?",
        ]
        
        safe_topics = [
            "the importance of education",
            "how to grow a garden",
            "the history of art",
            "basic first aid",
            "time management skills",
            "the scientific method",
            "different cultures around the world",
            "the importance of sleep",
            "how to learn a new language",
            "the basics of cooking",
        ]
        
        for template in hard_negative_templates:
            for topic in safe_topics:
                prompts.append({
                    "prompt": template.format(topic=topic),
                    "source": "hard_negative_generated",
                    "category": "hard_negative",
                })
                if len(prompts) >= n_prompts:
                    break
            if len(prompts) >= n_prompts:
                break
    
    return prompts[:n_prompts]


def generate_activation_multilayer_dataset(
    extractor: BaseModelActivationExtractor,
    n_jailbreak: int = 500,
    n_safe: int = 500,
    layer_indices: Optional[List[int]] = None,
    position: str = "last",
    save_path: Optional[str] = None,
) -> List[ExampleDict]:
    """
    Generate dataset with extracted activations from base LLM.
    
    Args:
        extractor: BaseModelActivationExtractor instance.
        n_jailbreak: Number of jailbreak examples to generate.
        n_safe: Number of safe examples to generate.
        layer_indices: Which layers to extract from (uses extractor's default if None).
        position: Token position to extract ("last", "all", or int).
        save_path: Optional path to save the dataset.
    
    Returns:
        List of examples in dict form.
    """
    if layer_indices is None:
        layer_indices = extractor.layer_indices
    
    dataset: List[ExampleDict] = []
    
    # Load jailbreak prompts
    print(f"Loading {n_jailbreak} jailbreak prompts...")
    jailbreak_prompts = load_jailbreak_prompts(max_examples=n_jailbreak)
    
    # Generate safe prompts
    print(f"Generating {n_safe} safe prompts...")
    safe_prompts = generate_safe_prompts(n_prompts=n_safe)
    
    # Process jailbreak examples
    print("Extracting activations for jailbreak examples...")
    for i, prompt_dict in enumerate(jailbreak_prompts[:n_jailbreak]):
        if i % 50 == 0:
            print(f"  Processed {i}/{n_jailbreak} jailbreak examples")
        
        activations = extractor.extract_activations(
            prompt=prompt_dict["prompt"],
            layer_indices=layer_indices,
            position=position,
        )
        
        # Convert tensors to lists for serialization
        activations_list = {
            layer: acts.tolist() for layer, acts in activations.items()
        }
        
        example = MultiLayerExample(
            activations_by_layer=activations_list,
            label="JAILBREAK",
            prompt=prompt_dict["prompt"],
            prompt_source=prompt_dict["source"],
            model_name=extractor.model_name,
            meta={
                "category": prompt_dict.get("category", "unknown"),
                "position": position,
            },
        )
        dataset.append(example.to_dict())
    
    # Process safe examples
    print("Extracting activations for safe examples...")
    for i, prompt_dict in enumerate(safe_prompts[:n_safe]):
        if i % 50 == 0:
            print(f"  Processed {i}/{n_safe} safe examples")
        
        activations = extractor.extract_activations(
            prompt=prompt_dict["prompt"],
            layer_indices=layer_indices,
            position=position,
        )
        
        # Convert tensors to lists for serialization
        activations_list = {
            layer: acts.tolist() for layer, acts in activations.items()
        }
        
        example = MultiLayerExample(
            activations_by_layer=activations_list,
            label="SAFE",
            prompt=prompt_dict["prompt"],
            prompt_source=prompt_dict["source"],
            model_name=extractor.model_name,
            meta={
                "category": prompt_dict.get("category", "unknown"),
                "position": position,
            },
        )
        dataset.append(example.to_dict())
    
    print(f"Generated {len(dataset)} total examples.")
    
    # Save if path provided
    if save_path:
        save_dataset(dataset, save_path)
        print(f"Dataset saved to {save_path}")
    
    return dataset


def save_dataset(
    dataset: List[ExampleDict],
    path: str,
    format: str = "json",
) -> None:
    """
    Save dataset to disk.
    
    Args:
        dataset: List of examples.
        path: File path to save to.
        format: "json" or "pickle".
    """
    path_obj = Path(path)
    path_obj.parent.mkdir(parents=True, exist_ok=True)
    
    if format == "json":
        with open(path, 'w') as f:
            json.dump(dataset, f, indent=2)
    elif format == "pickle":
        with open(path, 'wb') as f:
            pickle.dump(dataset, f)
    else:
        raise ValueError(f"Unsupported format: {format}")


def load_dataset(
    path: str,
    format: str = "json",
) -> List[ExampleDict]:
    """
    Load dataset from disk.
    
    Args:
        path: File path to load from.
        format: "json" or "pickle".
    
    Returns:
        List of examples.
    """
    if format == "json":
        with open(path, 'r') as f:
            return json.load(f)
    elif format == "pickle":
        with open(path, 'rb') as f:
            return pickle.load(f)
    else:
        raise ValueError(f"Unsupported format: {format}")


if __name__ == "__main__":
    import sys
    
    print("=" * 60)
    print("Module 1: Multi-Layer Jailbreak Dataset Generation")
    print("=" * 60)
    
    # Test activation extraction (requires model download)
    if "--activation" in sys.argv:
        print("\nGenerating dataset with extracted activations...")
        print("  (This will download the base model if not cached)")
        
        # Initialize extractor
        extractor = BaseModelActivationExtractor(
            model_name="google/gemma-2-2b-it",
            layer_indices=[7, 12, 23],
        )
        
        # Generate small dataset with extracted activations
        activation_data = generate_activation_multilayer_dataset(
            extractor=extractor,
            n_jailbreak=5,
            n_safe=5,
            save_path="data/activation_dataset_demo.json",
        )
        
        print(f"✓ Generated {len(activation_data)} examples with extracted activations")
        print(f"  Saved to: data/activation_dataset_demo.json")
        
        # Show activation shapes
        example = activation_data[0]
        print(f"\n  Example activation shapes:")
        for layer, acts in example['activations_by_layer'].items():
            print(f"    Layer {layer}: {len(acts)} positions × {len(acts[0])} dims")
    else:
        print("\n[Test 2] Skipped (use --activation flag to test activation extraction)")
        print("  Example: python src/data.py --activation")
    
    print("\n" + "=" * 60)
    print("Module 1: Complete")
    print("=" * 60)
