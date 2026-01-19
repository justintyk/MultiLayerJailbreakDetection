"""
Module 2: Models - Load Karvonen Activation Oracle and define multi-layer interface
"""

from typing import Dict, List, Tuple

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from peft import PeftConfig, PeftModel

# default AO checkpoint from Karvonen's HF collection
DEFAULT_AO_NAME = "adamkarvonen/checkpoints_cls_latentqa_past_lens_gemma-3-1b-it"


def load_activation_oracle(
    model_name: str = DEFAULT_AO_NAME,
) -> Tuple[AutoTokenizer, AutoModelForCausalLM]:
    """
    Load a pretrained Activation Oracle model and tokenizer from Hugging Face.

    This follows the Activation Oracles paper setup, where the AO is a causal LM
    that consumes an oracle prompt with placeholder tokens and injected activations.
    
    Args:
        model_name: Hugging Face model id for the Activation Oracle.

    Returns:
        tokenizer: HF tokenizer for the AO.
        model:     HF causal LM for the AO, moved to CUDA device.
    """
    device = torch.device("cuda")

    # Get base model name
    peft_cfg = PeftConfig.from_pretrained(model_name)
    base_name = peft_cfg.base_model_name_or_path

    # Load tokenizer & model
    tokenizer = AutoTokenizer.from_pretrained(base_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base_model = AutoModelForCausalLM.from_pretrained(
        base_name,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
    )

    model = PeftModel.from_pretrained(base_model, model_name)
    model.to(device)
    model.eval()
    return tokenizer, model

class MultiLayerActivationOracle:
    """
    Wrapper around a Karvonen Activation Oracle that adds a clean interface for
    multi-layer activation inputs.

    This is the main entry point for our novel method: we aggregate activations
    from multiple layers of the base model and query the AO for safety-relevant
    judgments (e.g., JAILBREAK vs SAFE).
    """

    def __init__(
        self,
        model_name: str = DEFAULT_AO_NAME,
        placeholder_token: str = " ?",
    ) -> None:
        self.tokenizer, self.model = load_activation_oracle(model_name)
        self.device = torch.device("cuda")
        self.placeholder_token = placeholder_token

    def _encode_multi_layer_activations(
        self,
        activations_by_layer: Dict[int, torch.Tensor],
        strategy: str = "mean",
    ) -> torch.Tensor:
        """
        Aggregate activations from multiple layers into a single tensor.

        Args:
            activations_by_layer: dict[layer -> tensor[T, D]]
              - T = number of token positions you saved for that layer
              - D = hidden_dim
            strategy: Aggregation strategy - "mean", "concat", or "last"
              - "mean": Average activations across token positions per layer [K, D]
              - "concat": Concatenate all layers into single vector [1, K*D]
              - "last": Use only last token position per layer [K, D]

        Returns:
            aggregated: Tensor to be injected at placeholder positions.
                       Shape depends on strategy:
                       - "mean"/"last": [K, D] where K = num layers
                       - "concat": [1, K*D]

        NOTE:
            This function is where our multi-layer representation lives and is
            the key technical novelty of the project. Different strategies allow
            us to explore how distributed patterns across depth are best encoded.
        """
        if not activations_by_layer:
            raise ValueError("No activations provided to MultiLayerActivationOracle.")

        if strategy == "mean":
            # Average over token positions, keep separate per layer
            mean_acts = []
            for layer in sorted(activations_by_layer.keys()):
                acts = activations_by_layer[layer]  # [T, D]            
                mean_acts.append(acts.mean(dim=0))  # [D]
            aggregated = torch.stack(mean_acts, dim=0)  # [K, D]
            
        elif strategy == "last":
            # Use only last token position per layer
            last_acts = []
            for layer in sorted(activations_by_layer.keys()):
                acts = activations_by_layer[layer]  # [T, D]
                last_acts.append(acts[-1])  # [D]
            aggregated = torch.stack(last_acts, dim=0)  # [K, D]
            
        elif strategy == "concat":
            # Concatenate all layers into single vector
            concat_acts = []
            for layer in sorted(activations_by_layer.keys()):
                acts = activations_by_layer[layer]  # [T, D]
                concat_acts.append(acts.mean(dim=0))  # [D]
            aggregated = torch.cat(concat_acts, dim=0).unsqueeze(0)  # [1, K*D]
            
        else:
            raise ValueError(f"Unknown aggregation strategy: {strategy}")

        return aggregated.to(self.device)

    def build_oracle_prompt(
        self,
        layer_description: str,
        num_placeholders: int,
        question: str,
    ) -> str:
        """
        Construct the oracle prompt of the form:
            'Layers: <desc> <ACT> <ACT> ... <ACT> QUESTION'

        Following the Activation Oracles input structure, we use placeholder
        tokens that will be associated with injected activation vectors.

        Args:
            layer_description: e.g. 'L7+L12+L23' or a more detailed text.
            num_placeholders:  number of activation vectors we plan to inject.
            question:          natural-language question, e.g.
                               'Is this a jailbreak attempt?'

        Returns:
            A string AO prompt to be tokenized and used with activation injection.
        """
        placeholders = " ".join([self.placeholder_token] * num_placeholders)
        prompt = f"Layers {layer_description}: {placeholders} {question}"
        return prompt

    def predict_label(
        self,
        activations_by_layer: Dict[int, torch.Tensor],
        question: str,
        layer_description: str = "multi-layer",
        max_new_tokens: int = 16,
    ) -> str:
        """
        High-level API: given multi-layer activations and a natural-language
        question, query the Activation Oracle and return its decoded answer.

        In our project, 'question' will typically be something like:
            'Is this activation pattern a distributed jailbreak? Answer YES or NO.'

        Args:
            activations_by_layer: dict[layer_index -> tensor[T, D]]
            question:            natural-language question about the activations.
            layer_description:   textual description of which layers are used.
            max_new_tokens:      decoding length for the AO.

        Returns:
            Decoded answer string from the AO (e.g. 'YES' / 'NO').
        """
        # 1. Aggregate multi-layer activations into K vectors.
        agg_acts = self._encode_multi_layer_activations(activations_by_layer)  # [K, D]
        K = agg_acts.shape[0]

        # 2. Build Oracle prompt with K placeholder positions.
        prompt = self.build_oracle_prompt(
            layer_description=layer_description,
            num_placeholders=K,
            question=question,
        )

        # 3. Tokenize prompt.
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
        ).to(self.device)
        
        input_ids = inputs["input_ids"]  # [1, seq_len]
        
        # 4. Find placeholder token positions
        placeholder_token_id = self.tokenizer.encode(
            self.placeholder_token, 
            add_special_tokens=False
        )[0]
        
        placeholder_positions = (input_ids[0] == placeholder_token_id).nonzero(as_tuple=True)[0]
        
        if len(placeholder_positions) != K:
            raise ValueError(
                f"Expected {K} placeholder tokens but found {len(placeholder_positions)}. "
                f"Prompt: {prompt}"
            )
        
        # 5. Get embeddings and inject activations
        with torch.no_grad():
            # Get the embedding layer
            if hasattr(self.model, 'get_input_embeddings'):
                embed_layer = self.model.get_input_embeddings()
            elif hasattr(self.model.base_model, 'get_input_embeddings'):
                embed_layer = self.model.base_model.get_input_embeddings()
            else:
                raise AttributeError("Cannot find embedding layer in model")
            
            # Get embeddings for all tokens
            embeddings = embed_layer(input_ids)  # [1, seq_len, D]
            
            # Replace placeholder embeddings with activation vectors
            for i, pos in enumerate(placeholder_positions):
                embeddings[0, pos, :] = agg_acts[i, :]
            
            # 6. Forward pass with injected activations
            output_ids = self.model.generate(
                inputs_embeds=embeddings,
                max_new_tokens=max_new_tokens,
                attention_mask=inputs.get("attention_mask"),
            )

        full_text = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)
        answer = full_text[len(prompt):].strip()  # naive post-processing
        return answer


if __name__ == "__main__":
    oracle = MultiLayerActivationOracle()
    print("Module 2: Activation Oracle wrapper initialized.")
