# MultiLayerJailbreakDetection

Detects jailbreaks across multiple neural network layers using Activation Oracles

## Abstract
This work investigates whether a single high-level behavioral concept (e.g., a jailbreak behavior) corresponds to multiple distinct activation patterns inside a large language model (LLM), and aims to learn concept-conditioned mappings from natural-language descriptions to activation-level interventions that can elicit and explore these patterns. We train an agent that takes a natural-language concept and outputs activation-level interventions for a target LLM, enabling us to probe the structure of internal representations underlying that behavior.

## Contents
1. [Setup](#1-setup)
2. [Dataset](#2-dataset)
3. [Models](#3-models)
4. [Training](#4-training)

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

```bash
# Run activation extraction pipeline (requires GPU and model download)
python src/data.py --activation
```

## 3. Models

The `MultiLayerActivationOracle` wraps a pretrained Activation Oracle model and provides an interface for multi-layer activation inputs.

### Usage

Load and query the oracle:

```python
from src.models import MultiLayerActivationOracle

# Initialize oracle (loads pretrained model from HuggingFace)
oracle = MultiLayerActivationOracle()

# Query with multi-layer activations
activations_by_layer = {
    7: torch.tensor([[0.1, 0.2, ...]]),   # [seq_len, hidden_dim]
    12: torch.tensor([[0.3, 0.4, ...]]),
    23: torch.tensor([[0.5, 0.6, ...]])
}

# Get natural language prediction
answer = oracle.predict_label(
    activations_by_layer=activations_by_layer,
    question="Is this a jailbreak attempt?",
    layer_description="L7+L12+L23"
)
print(answer)  # e.g., "YES" or "NO"
```

Load a custom oracle model:

```python
from src.models import load_activation_oracle

# Load specific checkpoint
tokenizer, model = load_activation_oracle(
    model_name="adamkarvonen/checkpoints_cls_latentqa_past_lens_gemma-3-1b-it"
)
```

Command line usage:

```bash
# Initialize and load activation oracle
python src/models.py
```

## 4. Training

Train the Activation Oracle on multi-layer jailbreak detection using full activation-conditioned SFT.

### Basic Training

Binary classification (SAFE vs JAILBREAK):

```bash
python src/train.py --n_examples 1000
```

### Programmatic Usage

```python
from src.train import train_full_oracle_sft

# Train with default settings
trainer = train_full_oracle_sft(
    n_examples=1000,
    batch_size=8,
    num_epochs=3,
    learning_rate=2e-5,
    use_extended_taxonomy=False,
    output_dir="./oracle_sft_output"
)
```