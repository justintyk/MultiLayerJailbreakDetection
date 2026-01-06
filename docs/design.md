Design, Dependencies, and Novelty

This document specifies (1) which external code and models we build on and links to them, (2) how those resources are used in this repository, and (3) which components and ideas are novel to our MultiLayerJailbreakDetection project.

## 1. External code and models we build on

### 1.1 LatentQA: Teaching LLMs to Decode Activations Into Natural Language

We draw heavily on the LatentQA activation-decoding framework:

- Project/paper:  
  LatentQA: Teaching LLMs to Decode Activations Into Natural Language.
- Code repository:  
  https://github.com/aypan17/latentqa

LatentQA introduces **Latent Interpretation Tuning (LIT)**: a method for finetuning a decoder LLM on a dataset of activations and associated question–answer pairs, so that the decoder learns to *read* model activations in natural language and, in some settings, to *control* behavior via activation edits.

The key architectural ideas we adopt are:

- Extract activations from a **target LLM** at specified layers during a forward pass.
- Patch those activations into a **decoder LLM** (often a copy of the target).
- Train the decoder by **supervised fine-tuning** on activation-conditioned examples (activations + question → answer).

### 1.2 Activation Oracles (Karvonen et al.)

We also build on Adam Karvonen’s Activation Oracle work, which operationalizes the idea of *Activation Oracles*—models trained to decode and intervene on activations:

- **Activation Oracle models (Hugging Face collection)**  
  https://huggingface.co/collections/adamkarvonen/activation-oracles  
  This collection hosts pretrained Activation Oracle models that map activations to natural-language descriptions, judgments, or control signals.

- **Activation Oracle demo notebook**  
  https://github.com/adamkarvonen/activation_oracles/blob/main/experiments/activation_oracle_demo.ipynb  
  This notebook demonstrates how to:
  - load an Activation Oracle,
  - prepare activation inputs in the expected format,
  - query the oracle for textual outputs and perform simple experiments over the activation space.

Karvonen’s public communication (e.g., tweets and references such as the linked X post) positions Activation Oracles as a practical, code-backed instantiation of the broader activation-decoding idea.

Together, **LatentQA** and **Activation Oracles** provide:

- A **conceptual template** for activation-decoding (activation extraction + decoder + SFT).
- **Concrete models and examples** for mapping internal activations to natural-language outputs and control signals.

We explicitly stand on this foundation.

---

## 2. How this repository uses those resources

Our project adopts the LatentQA / Activation Oracle paradigm as infrastructure and repurposes it for a new safety task: **detecting distributed jailbreaks from multi-layer activations**.

### 2.1 Architectural reuse

At the architectural level, we directly reuse the following patterns:

1. **Activation extraction from a base LLM**

   - We follow the LatentQA and Activation Oracle practice of:
     - registering hooks on a base model (e.g., a Gemma / Llama family model),
     - capturing activations at one or more layers for a given prompt or interaction.
   - This defines the “activation interface” between the base model and our detector.

2. **Activation-patched decoder / oracle**

   - In LatentQA, a decoder LLM is trained to consume activations (via a pseudo-token such as `[Act]`) and answer natural-language questions about them.
   - In Activation Oracles, a smaller oracle model is trained to read activations and output interpretable text or judgments.
   - We adopt the same pattern:
     - a **decoder / oracle model** that takes as input a representation of activations,
     - a **patched forward pass** that injects activations into the model’s computation,
     - a **supervised objective** defined on outputs conditioned on those activations.

3. **Supervised fine-tuning loop**

   - Like LatentQA’s LIT, we use standard supervised fine-tuning to train the decoder/oracle:
     - datasets of (activations, metadata, label),
     - cross-entropy loss on the desired label (textual or categorical),
     - standard optimization setups (AdamW, LR schedules).
   - This machinery is treated as **established infrastructure**, not a contribution of this repository.

### 2.2 Module-level mapping in our repo

Concretely, the mapping of these ideas into our codebase is:

- **`src/models.py`**
  - Implements a **decoder / oracle model** that accepts activation inputs from a base LLM.
  - The high-level interface mirrors the LatentQA / Activation Oracle pattern: load a model (e.g., an Activation Oracle checkpoint from the Hugging Face collection), define how activations are passed in, and expose a `forward` / `generate` method for downstream use.

- **`src/train.py`**
  - Implements a **supervised fine-tuning pipeline** analogous to Latent Interpretation Tuning:
    - read activation-annotated examples,
    - condition the decoder/oracle on those activations,
    - optimize a loss over desired labels (in our case, jailbreak vs safe, rather than open-ended QA).
      
- **`src/data.py` and `src/eval.py`**
  - Mirror LatentQA/Activation Oracle practice of:
    - constructing datasets of activation + label pairs,
    - evaluating decoder/oracle performance on held-out activations,
    - summarizing metrics (accuracy, calibration, etc.).
      
In summary: **we adopt the activation-decoding stack (LatentQA + Activation Oracles) as our implementation substrate and change the task, input representation, and evaluation regime.**

---

## 3. What we need to build and what is novel

This section separates **engineering work** we must do in this repository from the **scientific/technical novelty** we claim.

### 3.1 Implementation work specific to this project

On top of the external code and models above, this repository must implement:

1. **Multi-layer jailbreak dataset (`src/data.py`)**

   - A dataset of examples of the form:
     - underlying prompt / interaction trace,
     - activations from *multiple* layers of the base LLM (e.g., early, middle, late layers),
     - a label indicating whether the interaction constitutes a **jailbreak** (safety bypass) or is **safe**, potentially with a richer taxonomy of jailbreak modes.
     - defining how and where activations are recorded,
     - curating or generating jailbreak vs safe prompts,
     - tooling to serialize these as training/evaluation splits.

2. **Multi-layer activation encoder (`src/models.py`)**

   - Mechanisms to **aggregate activations from multiple layers** into a single representation usable by the decoder/oracle:
     - e.g., concatenation, pooling across layers, learned linear mixing, or attention over layer indices.
   - This extends prior single-layer activation-decoding setups to explicitly handle **distributed patterns across depth**.

3. **Jailbreak-detection training objective (`src/train.py`)**

   - A training objective specialized for **jailbreak detection**, rather than generic QA:
     - typically cross-entropy over labels such as `{SAFE, JAILBREAK}` or a small set of interpretable jailbreak types.
   - Integration of this objective into a LatentQA/Activation Oracle style training loop:
     - load multi-layer activation examples,
     - condition the decoder/oracle appropriately,
     - optimize detection performance.

4. **Safety-oriented evaluation and reporting (`src/eval.py`, `src/presentation.py`)**

   - Evaluation code that:
     - compares **single-layer** vs **multi-layer** detectors,
     - quantifies performance on realistic jailbreak attempts and closely matched safe prompts (hard negatives),
     - exports results in tabular and figure formats for analysis and communication.
   - Reporting code that highlights safety-relevant metrics (e.g., jailbreak recall at fixed false-positive rate).



