# MultiLayerJailbreakDetection
Detects jailbreaks across multiple neural network layers using Activation Oracles

# Goal
This project studies how jailbreak behavior arises from distributed activation patterns across multiple layers of a language model. We build on LatentQA-style activation→text decoding and Activation Oracles to detect and explain multi-layer jailbreak circuits in natural language. 

key papers:
- LatentQA: https://huggingface.co/papers/2412.08686
- Activation Oracles: https://huggingface.co/collections/adamkarvonen/activation-oracles

5 Modules:
1. Data generation (JSON datasets)
2. Model loading (Karvonen AO)
3. Training (TRL SFT)
4. Evaluation (92% accuracy target)
5. Slides (PPT-ready CSV)

