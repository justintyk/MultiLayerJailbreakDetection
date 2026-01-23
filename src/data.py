"""
Module 1: Data & Tasks - Generate multi-layer jailbreak dataset.

This module defines the data schema and generators for our multi-layer 
activation-level jailbreak detection task.

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

# Import rubric types from rubrics.py (handle both absolute and relative imports)
try:
    from src.rubrics import RubricDefinition
except ImportError:
    try:
        # Try relative import if running as module
        from .rubrics import RubricDefinition
    except ImportError:
        # Fallback: try direct import if in same directory
        try:
            import sys
            from pathlib import Path
            # Add parent directory to path if running directly
            sys.path.insert(0, str(Path(__file__).parent.parent))
            from src.rubrics import RubricDefinition
        except ImportError:
            # Last resort: define a minimal stub
            RubricDefinition = None

Label = Literal["JAILBREAK", "SAFE"]

"""
Each example in the dataset has the following structure:
    - activations_by_layer: dict[layer_idx -> tensor-like list of floats]
    - label:                "JAILBREAK" or "SAFE"
    - prompt:               the actual text prompt that generated activations
    - meta:                 optional metadata (prompt, source, model, etc.)
"""
class ExampleDict(TypedDict):
    activations_by_layer: Dict[int, List[List[float]]]
    label: Label
    meta: Dict[str, Any]
    prompt: str
    rubric: Optional[Dict[str, str]]  # NEW: {rubric_text, target_answer, category}


@dataclass
class MultiLayerExample:
    """
    Structured representation of one dataset example.
    - activations_by_layer: maps each layer index (e.g., 7, 12, 23) to a
      list of activation vectors extracted from the model.
    - label: "JAILBREAK" or "SAFE".
    - prompt: the actual text prompt that generated these activations.
    - meta: optional metadata (prompt_source, model_name, position, etc.)
    """
    activations_by_layer: Dict[int, List[List[float]]]
    label: Label
    prompt: str
    meta: Dict[str, Any]

    def to_dict(self) -> ExampleDict:
        return {
            "activations_by_layer": self.activations_by_layer,
            "label": self.label,
            "prompt": self.prompt,
            "meta": self.meta,
            "rubric": None,  # Not used in basic MultiLayerExample
        }


@dataclass
class RubricExample:
    """
    Dataset example with rubric-based labeling for policy training.
    
    Following (x, R, a) structure from arXiv:2502.01236:
    - prompt: x - Input prompt (neutral prompt)
    - rubric_text: R - Behavior description
    - target_answer: a - Property indicating satisfaction
    - response: y - Model response to prompt
    - satisfies_rubric: Ground truth label
    - verifier_score: p_v(a | R, y) score
    - activations_by_layer: Base activations f(x) from target model
    - meta: Additional metadata
    """
    prompt: str
    rubric_text: str
    target_answer: str
    category: str
    response: str
    satisfies_rubric: bool
    verifier_score: float
    activations_by_layer: Dict[int, torch.Tensor]
    meta: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        # Convert tensor activations to lists
        activations_list = {
            layer: acts.tolist() if isinstance(acts, torch.Tensor) else acts
            for layer, acts in self.activations_by_layer.items()
        }
        
        return {
            "prompt": self.prompt,
            "response": self.response,
            "satisfies_rubric": self.satisfies_rubric,
            "verifier_score": self.verifier_score,
            "activations_by_layer": activations_list,
            "meta": self.meta,
            "rubric": {
                "rubric_text": self.rubric_text,
                "target_answer": self.target_answer,
                "category": self.category,
            }
        }


class BaseModelActivationExtractor:
    """
    Extract activations from a base LLM at specified layers.
    
    This class loads a base model and registers forward hooks to capture hidden
    states at specified layers during inference.
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
                          If None, automatically calculates layers at 10%, 20%, 30%, 40%, 50% depth.
        """
        self.model_name = model_name
        self.device = torch.device("cuda")
        
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
        
        # Get total number of layers
        if hasattr(self.model, 'model') and hasattr(self.model.model, 'layers'):
            num_layers = len(self.model.model.layers)  # Gemma, Llama structure
        elif hasattr(self.model, 'transformer') and hasattr(self.model.transformer, 'h'):
            num_layers = len(self.model.transformer.h)  # GPT-2 structure
        else:
            raise ValueError(f"Unsupported model architecture: {self.model_name}")
        
        # Calculate layer indices at 25%, 50%, 75% depth if not provided
        if layer_indices is None:
            depth_percentages = [0.25, 0.5, 0.75]
            layer_indices = [int(num_layers * pct) for pct in depth_percentages]
            print(f"Auto-calculated layer indices for {model_name} ({num_layers} layers): {layer_indices}")
        
        self.layer_indices = layer_indices
        
        # Store activations
        self.activations: Dict[int, torch.Tensor] = {}
        self.hooks = []
        
    def _register_hooks(self, layer_indices: List[int]) -> None:
        """Register forward hooks to capture activations on specified layers."""
        self._clear_hooks()
        self.activations = {}
        
        # Get the model's layers
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
            # We assume batch_size = 1 (one prompt at a time)
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


def load_rubric_triples(
    triples_path: str = "data/rubric_triples.json",
) -> List[Dict[str, Any]]:
    """
    Load (x, R, a) triples from JSON file generated by rubrics.py.
    
    Args:
        triples_path: Path to rubric triples JSON file.
    
    Returns:
        List of dicts with keys: "prompt" (x), "rubric_text" (R), 
        "target_answer" (a), "metadata".
    """
    if not Path(triples_path).exists():
        raise FileNotFoundError(
            f"Rubric triples file not found at {triples_path}. "
            "Run 'python src/rubrics.py --dataset <name> --num-samples <n>' first."
        )
    
    with open(triples_path, "r", encoding="utf-8") as f:
        triples = json.load(f)
    
    if not triples:
        raise ValueError(f"No triples found in {triples_path}")
    
    return triples


# DEPRECATED: Use rubrics.py to generate (x, R, a) triples instead.
def load_jailbreak_prompts(
    max_examples: Optional[int] = None,
) -> List[Dict[str, str]]:
    """
    [DEPRECATED] Load jailbreak prompts from JailbreakBench dataset.
    
    This function is deprecated. Use rubrics.py to generate (x, R, a) triples
    from JailbreakBench goals instead.
    
    Args:
        max_examples: Maximum number of examples to load (None = all).
    
    Returns:
        List of dicts with keys: "prompt", "source".
    """
    import warnings
    warnings.warn(
        "load_jailbreak_prompts() is deprecated. Use rubrics.py to generate "
        "jailbreak rubrics and (x, R, a) triples instead.",
        DeprecationWarning,
        stacklevel=2
    )
    
    try:
        import jailbreakbench as jbb
        
        # Load JailbreakBench dataset
        dataset = jbb.read_dataset()
        goals = dataset.goals  # The actual jailbreak prompts
        
        # Store jailbreak prompts
        prompts = []
        for goal in goals:            
            prompts.append({
                "prompt": goal,  # jailbreak prompt
                "source": "jailbreakbench",
            })
            
            if max_examples and len(prompts) >= max_examples:
                break
                
    except ImportError:
        raise ImportError(
            "jailbreakbench package not installed. "
            "Install with: pip install jailbreakbench"
        )
    
    return prompts


# DEPRECATED: Use rubrics.py to load neutral prompts from WildChat/UltraChat instead.
def generate_safe_prompts(
    n_prompts: int = 100,
    dataset_sources: List[str] = ["alpaca", "dolly"],
) -> List[Dict[str, str]]:
    """
    [DEPRECATED] Load safe prompts from instruction-following datasets (Alpaca, Dolly).
    
    This function is deprecated. Use rubrics.py to load neutral prompts from
    WildChat/UltraChat instead, which are included in (x, R, a) triples.
    
    Args:
        n_prompts: Number of safe prompts to load.
        dataset_sources: Which datasets to load from. Options: "alpaca", "dolly".
    
    Returns:
        List of dicts with keys: "prompt", "source".
    """
    import warnings
    warnings.warn(
        "generate_safe_prompts() is deprecated. Use rubrics.py to generate "
        "(x, R, a) triples with neutral prompts from WildChat/UltraChat instead.",
        DeprecationWarning,
        stacklevel=2
    )
    prompts = []
    
    # Load from Alpaca dataset (Stanford, 52k instruction-following examples)
    if "alpaca" in dataset_sources:
        try:
            from datasets import load_dataset
            
            # Load cleaned Alpaca dataset from HuggingFace
            alpaca_data = load_dataset("tatsu-lab/alpaca", split="train")
            
            # Sample from Alpaca (evenly across all safe datasets)
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
                
                # Combine instruction and context if context exists
                if context:
                    prompt = f"{instruction}\n\nContext: {context}"
                else:
                    prompt = instruction
                
                prompts.append({
                    "prompt": prompt,
                    "source": "dolly",
                })
                
                if len(prompts) >= n_prompts:
                    break
                    
        except ImportError:
            print("Warning: 'datasets' package not installed. Skipping Dolly dataset.")
            print("Install with: pip install datasets")
        except Exception as e:
            print(f"Warning: Could not load Dolly dataset: {e}")
    
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
            meta={
                "prompt_source": prompt_dict["source"],
                "model_name": extractor.model_name,
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
            meta={
                "prompt_source": prompt_dict["source"],
                "model_name": extractor.model_name,
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


def generate_rubric_dataset(
    extractor: BaseModelActivationExtractor,
    triples_path: str = "data/rubric_triples.json",
    verifier=None,  # Verifier from rubrics.py
    layer_indices: Optional[List[int]] = None,
    position: str = "last",
    max_examples: Optional[int] = None,
    verbose: bool = True,
) -> List[RubricExample]:
    """
    Generate rubric-based dataset for policy training from (x, R, a) triples.
    
    Loads (x, R, a) triples generated by rubrics.py and creates RubricExample objects by:
    1. Extracting base activations f(x) for each prompt x
    2. Generating response y from target model
    3. Verifying rubric satisfaction with verifier (if provided)
    4. Building RubricExample with all components
    
    Args:
        extractor: BaseModelActivationExtractor for target model
        triples_path: Path to (x, R, a) triples JSON file from rubrics.py
        verifier: Verifier instance for scoring responses (optional, from rubrics.py)
        layer_indices: Layers to extract activations from (uses extractor default if None)
        position: Token position for activation extraction ("last", "all", or int)
        max_examples: Maximum number of triples to process (None = all)
        verbose: Print progress information
        
    Returns:
        List of RubricExample objects with activations, responses, and verifier scores
    """
    if layer_indices is None:
        layer_indices = extractor.layer_indices
    
    # Load (x, R, a) triples from file
    if verbose:
        print(f"Loading (x, R, a) triples from {triples_path}...")
    triples = load_rubric_triples(triples_path)
    
    if max_examples:
        triples = triples[:max_examples]
    
    if verbose:
        print(f"Processing {len(triples)} triples...")
    
    dataset: List[RubricExample] = []
    
    for idx, triple in enumerate(triples):
        if verbose and (idx + 1) % 20 == 0:
            print(f"  Processed {idx + 1}/{len(triples)} examples")
        
        try:
            prompt = triple["prompt"]  # x
            rubric_text = triple["rubric_text"]  # R
            target_answer = triple["target_answer"]  # a
            metadata = triple.get("metadata", {})
            
            # 1. Extract base activations f(x)
            activations = extractor.extract_activations(
                prompt=prompt,
                layer_indices=layer_indices,
                position=position,
            )
            
            # 2. Generate response y from target model
            inputs = extractor.tokenizer(
                prompt,
                return_tensors="pt",
                padding=True,
                truncation=True,
            ).to(extractor.device)
            
            input_length = inputs["input_ids"].shape[1]
            
            with torch.no_grad():
                outputs = extractor.model.generate(
                    **inputs,
                    max_new_tokens=100,
                    do_sample=False,
                    pad_token_id=extractor.tokenizer.pad_token_id,
                )
            
            # Decode only the newly generated tokens (exclude the input prompt)
            response = extractor.tokenizer.decode(
                outputs[0][input_length:],
                skip_special_tokens=True
            )
            
            # 3. Verify rubric satisfaction (if verifier provided)
            if verifier is not None:
                if RubricDefinition is None:
                    raise ImportError("RubricDefinition could not be imported from src.rubrics")
                # Create RubricDefinition for verifier
                rubric_def = RubricDefinition(
                    rubric_text=rubric_text,
                    target_answer=target_answer,
                    category="jailbreak",  # All triples are jailbreak concepts
                    metadata=metadata,
                )
                score = verifier.verify(rubric_def, response)
                satisfies = score > 0.7  # Threshold for satisfaction
            else:
                # Default: assume neutral prompts don't satisfy jailbreak rubrics
                score = 0.0
                satisfies = False
            
            # 4. Create RubricExample
            example = RubricExample(
                prompt=prompt,
                rubric_text=rubric_text,
                target_answer=target_answer,
                category="jailbreak",  # All triples are jailbreak concepts
                response=response,
                satisfies_rubric=satisfies,
                verifier_score=score,
                activations_by_layer=activations,
                meta={
                    "model_name": extractor.model_name,
                    "position": position,
                    "source_dataset": metadata.get("source_dataset", "unknown"),
                    "example_index": metadata.get("example_index", idx),
                    "rubric_index": metadata.get("rubric_index", -1),
                }
            )
            
            dataset.append(example)
            
        except Exception as e:
            if verbose:
                print(f"  Error processing triple {idx}: {e}")
            continue
    
    if verbose:
        print(f"\nGenerated {len(dataset)} rubric examples total.")
    
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
    import argparse
    import sys
    
    parser = argparse.ArgumentParser(
        description="Process (x, R, a) triples and extract activations to build training dataset"
    )
    parser.add_argument(
        "--triples-path",
        type=str,
        default="data/rubric_triples.json",
        help="Path to (x, R, a) triples JSON file (default: data/rubric_triples.json)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/dataset.json",
        help="Output path for dataset with activations (default: data/dataset.json)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="google/gemma-2-2b-it",
        help="HuggingFace model name for activation extraction (default: google/gemma-2-2b-it)",
    )
    parser.add_argument(
        "--max-examples",
        type=int,
        default=None,
        help="Maximum number of triples to process (None = all)",
    )
    parser.add_argument(
        "--no-verifier",
        action="store_true",
        help="Skip verifier scoring (faster, but no verifier scores in output)",
    )
    
    args = parser.parse_args()
    
    # Load environment variables from .env file
    from dotenv import load_dotenv
    load_dotenv()
    
    print("=" * 60)
    print("Processing (x, R, a) Triples and Extracting Activations")
    print("=" * 60)
    print(f"Triples file:  {args.triples_path}")
    print(f"Output file:   {args.output}")
    print(f"Model:         {args.model}")
    if args.max_examples:
        print(f"Max examples:   {args.max_examples}")
    print("=" * 60)
    
    # Initialize extractor
    print("\n[1/3] Initializing activation extractor...")
    print(f"  (This will download {args.model} if not cached)")
    extractor = BaseModelActivationExtractor(model_name=args.model)
    print(f"  ✓ Extractor ready (layers: {extractor.layer_indices})")
    
    # Initialize verifier (optional)
    verifier = None
    if not args.no_verifier:
        print("\n[2/3] Initializing verifier...")
        try:
            # Try multiple import strategies
            try:
                from src.rubrics import Verifier
            except ImportError:
                try:
                    from .rubrics import Verifier
                except ImportError:
                    import sys
                    from pathlib import Path
                    sys.path.insert(0, str(Path(__file__).parent.parent))
                    from src.rubrics import Verifier
            
            verifier = Verifier(mode="llm")
            print("  ✓ Verifier ready (LLM-as-judge)")
        except Exception as e:
            print(f"  ⚠ Warning: Could not initialize verifier: {e}")
            print("  Continuing without verifier scoring...")
    
    # Process triples and extract activations
    print("\n[3/3] Processing triples and extracting activations...")
    examples = generate_rubric_dataset(
        extractor=extractor,
        triples_path=args.triples_path,
        verifier=verifier,
        max_examples=args.max_examples,
        verbose=True,
    )
    
    # Convert RubricExample objects to dicts for JSON serialization
    print(f"\nConverting {len(examples)} examples to JSON format...")
    dataset_dicts = [ex.to_dict() for ex in examples]
    
    # Save to file
    print(f"Saving dataset to {args.output}...")
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(dataset_dicts, f, indent=2, ensure_ascii=False)
    
    print("\n" + "=" * 60)
    print("Dataset Generation Complete")
    print("=" * 60)
    print(f"✓ Processed {len(examples)} examples")
    print(f"✓ Saved to: {args.output}")
    
