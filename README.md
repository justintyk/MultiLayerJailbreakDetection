# MultiLayerJailbreakDetection

Detects jailbreaks across multiple neural network layers using Activation Oracles

## Abstract
This work investigates whether a single high-level behavioral concept (e.g., a jailbreak behavior) corresponds to multiple distinct activation patterns inside a large language model (LLM), and aims to learn concept-conditioned mappings from natural-language descriptions to activation-level interventions that can elicit and explore these patterns. We train an agent that takes a natural-language concept and outputs activation-level interventions for a target LLM, enabling us to probe the structure of internal representations underlying that behavior.

## Contents
1. [Setup](#1-setup)
2. [Dataset](#2-dataset)

## 1. Setup
```bash
pip install -r requirements.txt
```

## 2. Dataset
Each example in the dataset has the following structure:

```python
{
    "activations_by_layer": {
        7: [[0.1, 0.2, ...]],   # List of activation vectors for layer 7
        12: [[0.3, 0.4, ...]],  # List of activation vectors for layer 12
        23: [[0.5, 0.6, ...]]   # List of activation vectors for layer 23
    },
    "label": "JAILBREAK" or "SAFE",
    "prompt": "The actual text prompt",
    "prompt_source": "jailbreakbench" or "safe_generated",
    "model_name": "google/gemma-2-2b-it",
    "meta": {
        "category": "harmful_category",
        "position": "last",
        ...
    }
}
```

Extract activations from prompt datasets:

```python
from src.data import BaseModelActivationExtractor, generate_activation_multilayer_dataset

# Initialize extractor
extractor = BaseModelActivationExtractor(
    model_name="google/gemma-2-2b-it",
    device="cuda",
    layer_indices=[7, 12, 23]
)

# Generate dataset with extracted activations
dataset = generate_activation_multilayer_dataset(
    extractor=extractor,
    n_jailbreak=500,
    n_safe=500,
    save_path="data/activation_dataset.json"
)
```

Load existing dataset:

```python
from src.data import load_dataset

# Load from JSON
dataset = load_dataset("data/activation_dataset.json", format="json")

# Load from pickle (faster for large datasets)
dataset = load_dataset("data/activation_dataset.pkl", format="pickle")
```

### Usage

Extract activations from custom prompts:

```python
from src.data import BaseModelActivationExtractor

extractor = BaseModelActivationExtractor(
    model_name="google/gemma-2-2b-it",
    device="cuda"
)

# Extract from a single prompt
activations = extractor.extract_activations(
    prompt="How do I make a bomb?",
    layer_indices=[7, 12, 23],
    position="last"  # "last", "all", or int
)

# activations is a dict: {layer_idx: tensor[num_positions, hidden_dim]}
print(f"Layer 7 shape: {activations[7].shape}")
```

Load jailbreak prompts:

```python
from src.data import load_jailbreak_prompts

# Load from JailbreakBench
prompts = load_jailbreak_prompts(
    dataset_name="jailbreakbench",
    max_examples=100
)

# Each prompt is a dict with keys: 'prompt', 'source', 'category'
for p in prompts[:3]:
    print(f"{p['category']}: {p['prompt'][:50]}...")
```

Load safe prompts:

```python
from src.data import generate_safe_prompts

# Load from Alpaca (52k examples) and Dolly (15k examples)
safe_prompts = generate_safe_prompts(
    n_prompts=100,
    dataset_sources=["alpaca", "dolly"],  # Choose which datasets to use
    include_hard_negatives=True
)

# Prompts are loaded from high-quality instruction-following datasets
for p in safe_prompts[:3]:
    print(f"{p['source']}: {p['prompt'][:50]}...")

# Use only Alpaca
alpaca_only = generate_safe_prompts(
    n_prompts=50,
    dataset_sources=["alpaca"]
)

# Use only Dolly
dolly_only = generate_safe_prompts(
    n_prompts=50,
    dataset_sources=["dolly"]
)
```

Command line usage:

```bash
# Run activation extraction pipeline (requires GPU and model download)
python src/data.py --activation
```