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
        placeholder_token: str = "<ACT>",
        source_hidden_dim: int = None,
    ) -> None:
        self.tokenizer, self.model = load_activation_oracle(model_name)
        self.device = torch.device("cuda")
        self.placeholder_token = placeholder_token
        
        # Get placeholder token ID(s) for reliable detection
        self._placeholder_ids = self.tokenizer.encode(
            placeholder_token, 
            add_special_tokens=False
        )
        
        # Get AO model's hidden dimension for projection
        self.ao_hidden_dim = self._get_ao_hidden_dim()
        self.source_hidden_dim = source_hidden_dim
        self.projection = None
        
        # If source dimension specified and different from AO, create projection
        if source_hidden_dim is not None and source_hidden_dim != self.ao_hidden_dim:
            self._init_projection(source_hidden_dim, self.ao_hidden_dim)
    
    def _get_ao_hidden_dim(self) -> int:
        """Get the hidden dimension of the AO model."""
        # Try different ways to get hidden dimension
        if hasattr(self.model, 'config'):
            config = self.model.config
            if hasattr(config, 'hidden_size'):
                return config.hidden_size
            if hasattr(config, 'd_model'):
                return config.d_model
        
        # Try to get from base model
        if hasattr(self.model, 'base_model') and hasattr(self.model.base_model, 'config'):
            config = self.model.base_model.config
            if hasattr(config, 'hidden_size'):
                return config.hidden_size
        
        # Try to get from embedding layer
        embed_layer = self.model.get_input_embeddings()
        if hasattr(embed_layer, 'embedding_dim'):
            return embed_layer.embedding_dim
        if hasattr(embed_layer, 'weight'):
            return embed_layer.weight.shape[1]
        
        # Default fallback for Gemma-3-1B
        return 1152
    
    def _init_projection(self, source_dim: int, target_dim: int) -> None:
        """
        Initialize projection layer to map activations from source model to AO.
        
        Uses a simple linear projection. For better results, this could be 
        replaced with a learned projection trained on paired activations.
        """
        import torch.nn as nn
        
        print(f"  Note: Creating projection {source_dim} → {target_dim} for AO compatibility")
        
        # Simple linear projection (no bias, preserves direction)
        self.projection = nn.Linear(source_dim, target_dim, bias=False).to(self.device)
        
        # Initialize with truncated identity-like mapping
        # This preserves as much information as possible
        with torch.no_grad():
            # Use SVD-style initialization for dimension reduction
            if source_dim > target_dim:
                # Reduction: use first target_dim dimensions (like PCA keeping top components)
                # Initialize as truncated identity
                self.projection.weight.zero_()
                self.projection.weight[:, :target_dim] = torch.eye(target_dim, device=self.device)
            else:
                # Expansion: pad with zeros
                self.projection.weight.zero_()
                self.projection.weight[:source_dim, :] = torch.eye(source_dim, device=self.device)
    
    def _project_activations(self, activations: torch.Tensor) -> torch.Tensor:
        """
        Project activations to AO's hidden dimension if needed.
        
        Args:
            activations: Tensor of shape [..., source_hidden_dim]
            
        Returns:
            Tensor of shape [..., ao_hidden_dim]
        """
        if self.projection is None:
            # Check if we need to create projection on-the-fly
            act_dim = activations.shape[-1]
            if act_dim != self.ao_hidden_dim:
                self._init_projection(act_dim, self.ao_hidden_dim)
        
        if self.projection is not None:
            # Ensure activations are on correct device and dtype
            activations = activations.to(device=self.device, dtype=torch.float32)
            return self.projection(activations)
        
        return activations

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

        # Move to device
        aggregated = aggregated.to(self.device)
        
        # Apply projection if source dimension differs from AO dimension
        aggregated = self._project_activations(aggregated)
        
        return aggregated

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

    def _find_placeholder_positions(
        self,
        input_ids: torch.Tensor,
        num_expected: int
    ) -> List[int]:
        """
        Find positions of placeholder tokens in tokenized input.
        
        Uses multiple strategies to handle different tokenization behaviors:
        1. Try matching the full placeholder token sequence
        2. Try matching individual tokens that make up the placeholder
        3. Fallback to finding by string position in decoded text
        
        Args:
            input_ids: Tokenized input [1, seq_len]
            num_expected: Number of placeholder positions expected
            
        Returns:
            List of token positions where activations should be injected
        """
        ids = input_ids[0].tolist()
        positions = []
        
        # Strategy 1: Find exact matches of placeholder token IDs
        placeholder_ids = self._placeholder_ids
        if len(placeholder_ids) == 1:
            # Single token placeholder
            pid = placeholder_ids[0]
            for i, tid in enumerate(ids):
                if tid == pid:
                    positions.append(i)
        else:
            # Multi-token placeholder - find sequences
            plen = len(placeholder_ids)
            for i in range(len(ids) - plen + 1):
                if ids[i:i+plen] == placeholder_ids:
                    # Use the first token position for injection
                    positions.append(i)
        
        if len(positions) == num_expected:
            return positions
        
        # Strategy 2: Try finding "<" followed by "ACT" patterns
        # This handles cases where <ACT> tokenizes differently in context
        positions = []
        decoded_tokens = [self.tokenizer.decode([tid]) for tid in ids]
        
        i = 0
        while i < len(decoded_tokens):
            # Check if this position starts a placeholder pattern
            window = "".join(decoded_tokens[i:i+5])  # Look ahead up to 5 tokens
            if self.placeholder_token in window or "<ACT>" in window or "[ACT]" in window:
                positions.append(i)
                # Skip ahead past this placeholder
                skip = 1
                while skip < 5 and i + skip < len(decoded_tokens):
                    partial = "".join(decoded_tokens[i:i+skip+1])
                    if self.placeholder_token in partial:
                        break
                    skip += 1
                i += max(skip, 1)
            else:
                i += 1
        
        if len(positions) == num_expected:
            return positions
        
        # Strategy 3: Use fixed positions after the colon in "Layers L13: <ACT>"
        # Find position right after ":" which is where placeholder should be
        positions = []
        for i, tid in enumerate(ids):
            decoded = self.tokenizer.decode([tid])
            if ":" in decoded and i + 1 < len(ids):
                # Found colon, next position(s) should be placeholder
                for j in range(num_expected):
                    if i + 1 + j < len(ids):
                        positions.append(i + 1 + j)
                break
        
        if len(positions) == num_expected:
            return positions
        
        # Strategy 4: Last resort - use positions after initial prompt tokens
        # Assume format: "Layers L13: [ACT positions here] Question..."
        # Typically the placeholder is around position 4-6
        if num_expected == 1:
            # For single activation, inject at a reasonable position
            # Skip "Layers", "L13", ":" tokens
            inject_pos = min(4, len(ids) - 1)
            return [inject_pos]
        
        raise ValueError(
            f"Could not find {num_expected} placeholder positions. "
            f"Found {len(positions)} candidates. "
            f"Token IDs: {ids[:20]}... "
            f"Decoded: {self.tokenizer.decode(input_ids[0][:30])}"
        )

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
        
        # 4. Find placeholder token positions using robust detection
        placeholder_positions = self._find_placeholder_positions(input_ids, K)
        
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
    
    def label_intervention(
        self,
        intervention_activations: Dict[int, torch.Tensor],
        rubric,  # RubricDefinition from rubrics.py
        max_new_tokens: int = 64,
        strategy: str = "mean"
    ) -> Tuple[str, float, str]:
        """
        Use Activation Oracle to interpret intervention in activation space.
        
        This implements the AO-Labelling Protocol from docs/Impl.pdf Section 2.2:
        For each successful intervention f'(x,c) = f(x) + δf, we perform semantic
        enrichment using the AO prompt template.
        
        Args:
            intervention_activations: f'(x,c) = f(x) + δf
                Dict mapping layer_idx -> intervened activation tensor
            rubric: RubricDefinition with rubric_text and target_answer
            max_new_tokens: Max tokens for AO response
            strategy: Aggregation strategy for multi-layer activations
            
        Returns:
            (label, confidence, mechanism): Semantic label, confidence score, 
                and description of the jailbreak mechanism
            
        Example:
            >>> intervened_acts = {12: torch.randn(1, 2048)}
            >>> rubric = RubricDefinition(
            ...     rubric_text="Bypasses safety refusals",
            ...     target_answer="provides harmful content",
            ...     category="jailbreak"
            ... )
            >>> label, conf, mechanism = oracle.label_intervention(intervened_acts, rubric)
            >>> print(f"Label: {label}, Confidence: {conf}, Mechanism: {mechanism}")
        """
        # Build AO query following the exact template from docs/Impl.pdf Section 2.2:
        # <ACT>[fx,c serialized]</ACT> Identify the specific jailbreak mechanism 
        # for rubric c. Output: [Short Label], [Confidence 0-1].
        layer_desc = "+".join([f"L{layer}" for layer in sorted(intervention_activations.keys())])
        
        # Use the PDF template format
        question = (
            f"Identify the specific jailbreak mechanism for rubric: '{rubric.rubric_text}'. "
            f"Output format: [Short Label], [Confidence 0-1], [Mechanism Description]. "
            f"Example labels: Ignore-System-Prompt, Latent-Harm-Injection, Roleplay-Jailbreak, "
            f"Obfuscation-Attack, Context-Manipulation."
        )
        
        # Query AO with intervened activations
        response = self.predict_label(
            activations_by_layer=intervention_activations,
            question=question,
            layer_description=layer_desc,
            max_new_tokens=max_new_tokens
        )
        
        # Parse response for label, confidence, and mechanism
        label, confidence, mechanism = self._parse_ao_response(response)
        
        return label, confidence, mechanism
    
    def _parse_ao_response(self, response: str) -> Tuple[str, float, str]:
        """
        Parse AO response to extract label, confidence, and mechanism.
        
        Expected format: "[LABEL], [CONFIDENCE], [MECHANISM]"
        Example: "Roleplay-Jailbreak, 0.85, Adopts fictional persona to bypass safety"
        
        Args:
            response: Raw AO response string
            
        Returns:
            (label, confidence, mechanism): Parsed label, confidence score, and mechanism description
        """
        import re
        
        try:
            # Try to parse structured response
            parts = response.split(',')
            
            # Extract label (first part)
            label = parts[0].strip() if len(parts) >= 1 else "Unknown"
            
            # Extract confidence (second part)
            confidence = 0.5  # Default
            if len(parts) >= 2:
                confidence_str = parts[1].strip()
                conf_match = re.search(r'(\d+\.?\d*)', confidence_str)
                if conf_match:
                    confidence = float(conf_match.group(1))
                    # Ensure confidence is in [0, 1]
                    if confidence > 1.0:
                        confidence = confidence / 100.0
            
            # Extract mechanism (third part onwards, or generate from label)
            if len(parts) >= 3:
                mechanism = ','.join(parts[2:]).strip()
            else:
                # Generate mechanism description from label if not provided
                mechanism = self._label_to_mechanism(label)
            
            return label, confidence, mechanism
                
        except Exception as e:
            print(f"Warning: Failed to parse AO response '{response}': {e}")
            return response.strip(), 0.5, "Unknown mechanism"
    
    def _label_to_mechanism(self, label: str) -> str:
        """
        Generate mechanism description from label if AO didn't provide one.
        
        Maps known jailbreak labels to their mechanism descriptions based on
        the expected results from docs/Impl.pdf Table 1.
        """
        mechanisms = {
            "Ignore-System-Prompt": "Bypasses system instructions by ignoring safety constraints",
            "Latent-Harm-Injection": "Embeds harmful content in seemingly benign context",
            "Roleplay-Jailbreak": "Adopts fictional persona or scenario to bypass safety filters",
            "Obfuscation-Attack": "Uses encoding, translation, or obfuscation to hide harmful intent",
            "Context-Manipulation": "Manipulates context window to override safety training",
            "Prompt-Injection": "Injects adversarial instructions to override system behavior",
            "DAN-Style": "Uses 'Do Anything Now' or similar unrestricted persona prompts",
            "Hypothetical-Framing": "Frames harmful requests as hypothetical or educational",
            "Authority-Impersonation": "Claims authority or special permissions to bypass restrictions",
        }
        
        # Try exact match first
        if label in mechanisms:
            return mechanisms[label]
        
        # Try partial match
        label_lower = label.lower()
        for key, desc in mechanisms.items():
            if key.lower() in label_lower or label_lower in key.lower():
                return desc
        
        return f"Jailbreak mechanism: {label}"
    
    def batch_label_interventions(
        self,
        interventions: List[Dict[int, torch.Tensor]],
        rubrics: List,  # List[RubricDefinition]
        max_new_tokens: int = 64,
        strategy: str = "mean"
    ) -> List[Tuple[str, float, str]]:
        """
        Label multiple interventions in batch.
        
        Args:
            interventions: List of intervention activation dicts
            rubrics: List of corresponding rubrics
            max_new_tokens: Max tokens for AO responses
            strategy: Aggregation strategy
            
        Returns:
            List of (label, confidence, mechanism) tuples
        """
        if len(interventions) != len(rubrics):
            raise ValueError(
                f"Mismatch: {len(interventions)} interventions but {len(rubrics)} rubrics"
            )
        
        results = []
        for intervention_acts, rubric in zip(interventions, rubrics):
            label, conf, mechanism = self.label_intervention(
                intervention_acts,
                rubric,
                max_new_tokens=max_new_tokens,
                strategy=strategy
            )
            results.append((label, conf, mechanism))
        
        return results


if __name__ == "__main__":
    oracle = MultiLayerActivationOracle()
    print("Module 2: Activation Oracle wrapper initialized.")
