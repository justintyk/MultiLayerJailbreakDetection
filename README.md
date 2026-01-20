# MultiLayerJailbreakDetection

An end-to-end Activation Oracle model for detecting jailbreaks across multiple neural network layers

## Abstract
This work investigates whether a single high-level behavioral concept (e.g., a jailbreak behavior) corresponds to multiple distinct activation patterns inside a large language model (LLM). We train an Activation Oracle that takes multi-layer activations from a target LLM and outputs natural-language safety assessments, enabling us to detect distributed jailbreak patterns that span multiple layers. Our approach uses activation-conditioned training following the LatentQA paradigm, where activation vectors are injected at placeholder positions during supervised fine-tuning.

## Contents
1. [Dataset](#1-dataset)
2. [Pre-trained Models](#2-pre-trained-models)
3. [Setup](#3-setup)
4. [Usage](#4-usage)

## 1. Dataset

Generate multi-layer activation dataset from jailbreak and safe prompts:

```bash
python src/data.py --activation
```
Dataset is saved to: `data/activation_dataset_demo.json`

## 2. Setup

Install dependencies:
```bash
pip install -r requirements.txt
```

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