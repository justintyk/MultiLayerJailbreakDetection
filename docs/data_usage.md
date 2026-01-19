# Quick Reference: data.py Usage

## Installation

```bash
# Install dependencies
pip install -r requirements.txt

# For real activation extraction, ensure torch and transformers are installed
pip install torch transformers jailbreakbench

# For loading Alpaca/Dolly datasets (optional, for safe prompts)
pip install datasets
```

## Basic Usage

### 1. Synthetic Dataset (Fast, for Testing)

```python
from src.data import generate_synthetic_multilayer_dataset, save_dataset

# Generate synthetic data
dataset = generate_synthetic_multilayer_dataset(
    n_examples=1000,
    layer_indices=[7, 12, 23],
    dim=4,
    p_jailbreak=0.5
)

# Save to file
save_dataset(dataset, "data/synthetic_dataset.json")
```

### 2. Real Activation Dataset (Production)

```python
from src.data import BaseModelActivationExtractor, generate_real_multilayer_dataset

# Initialize extractor
extractor = BaseModelActivationExtractor(
    model_name="google/gemma-2-2b-it",
    device="cuda",  # or "cpu"
    layer_indices=[7, 12, 23]
)

# Generate dataset with real activations
dataset = generate_real_multilayer_dataset(
    extractor=extractor,
    n_jailbreak=500,
    n_safe=500,
    save_path="data/real_dataset.json"
)
```

### 3. Load Existing Dataset

```python
from src.data import load_dataset

# Load from JSON
dataset = load_dataset("data/real_dataset.json", format="json")

# Load from pickle (faster for large datasets)
dataset = load_dataset("data/real_dataset.pkl", format="pickle")
```

## Advanced Usage

### Extract Activations from Custom Prompts

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

### Load Jailbreak Prompts

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

### Load Safe Prompts (Alpaca & Dolly)

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

## Command Line Usage

```bash
# Test synthetic dataset generation
python src/data.py

# Test real activation extraction (requires GPU/CPU and model download)
python src/data.py --real
```

## Dataset Schema

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
    "prompt_source": "jailbreakbench" or "safe_generated" or "synthetic",
    "model_name": "google/gemma-2-2b-it",
    "meta": {
        "category": "harmful_category",
        "position": "last",
        ...
    }
}
```

## Tips

1. **Start with synthetic data** for rapid prototyping and testing
2. **Use CPU for small experiments** (5-10 examples) to avoid GPU setup
3. **Use GPU for production datasets** (500+ examples) for faster extraction
4. **Save datasets in JSON** for human readability, **pickle for large datasets**
5. **Monitor memory usage** when extracting from many prompts at once

## Troubleshooting

### "No module named 'torch'"
```bash
pip install torch transformers
```

### "No module named 'jailbreakbench'"
```bash
pip install jailbreakbench
```

### CUDA out of memory
- Use smaller batch sizes
- Extract activations on CPU: `device="cpu"`
- Process prompts in smaller chunks

### Model download is slow
- Models are cached in `~/.cache/huggingface/`
- First download takes time, subsequent runs are fast
- Use smaller models for testing: `google/gemma-2-2b-it` (2B params)
