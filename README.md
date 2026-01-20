# MultiLayerJailbreakDetection

An end-to-end Activation Oracle model for detecting jailbreaks across multiple neural network layers

## Abstract
This work investigates whether a single high-level behavioral concept (e.g., a jailbreak behavior) corresponds to multiple distinct activation patterns inside a large language model (LLM). We train an Activation Oracle that takes multi-layer activations from a target LLM and outputs natural-language safety assessments, enabling us to detect distributed jailbreak patterns that span multiple layers. Our approach uses activation-conditioned training following the LatentQA paradigm, where activation vectors are injected at placeholder positions during supervised fine-tuning.

## Contents
1. [Setup](#1-setup)
2. [Dataset](#2-dataset)
3. [Usage](#3-usage)

## 1. Setup

Install dependencies:
```bash
pip install -r requirements.txt
```

## 2. Dataset

Generate multi-layer activation dataset from jailbreak and safe prompts:

```bash
python src/data.py --activation
```

### Configuration

To modify the dataset size or model configuration, edit the parameters in `src/data.py`:

```python
# Initialize extractor
extractor = BaseModelActivationExtractor(
    model_name="google/gemma-2-2b-it",
    layer_indices=[7, 12, 23],
)

# Generate dataset
activation_data = generate_activation_multilayer_dataset(
    extractor=extractor,
    n_jailbreak=5,
    n_safe=5,
    save_path="data/activation_dataset_demo.json",
)
```

Dataset is saved to `data/activation_dataset_demo.json`

## 3. Usage

### 3.1. Training

Train the oracle on multi-layer jailbreak detection:

```bash
# Binary classification
python src/train.py --n_examples 1000

# Extended taxonomy
python src/train.py --n_examples 2000 --extended_taxonomy
```
Model checkpoints saved to: `oracle_sft_output/`

## Citation

If you use this code in your research, please cite:

```bibtex
@misc{multilayerjailbreakdetection2026,
  title={Multi-Layer Jailbreak Detection using Activation Oracles},
  author={Justin Ku},
  year={2026},
  url={https://github.com/justintyk/MultiLayerJailbreakDetection}
}
```

## References

- [LatentQA: Teaching LLMs to Decode Activations Into Natural Language](https://github.com/aypan17/latentqa)
- [Activation Oracles](https://huggingface.co/collections/adamkarvonen/activation-oracles)
- [JailbreakBench](https://jailbreakbench.github.io/)