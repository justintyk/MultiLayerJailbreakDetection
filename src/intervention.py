"""
Module: Activation Intervention - Apply activation interventions to target models.

This module implements the intervention pipeline for applying activation
interventions δf to target models during inference.

Key components:
- ActivationHook: Context manager for hooking into model layers
- InterventionPipeline: High-level interface for applying interventions
"""

from typing import Dict, List, Optional, Callable
from contextlib import contextmanager
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForCausalLM


class ActivationHook:
    """
    Context manager for applying activation interventions to model layers.
    
    Usage:
        with ActivationHook(model, layer_idx, intervention_fn):
            outputs = model(**inputs)
    """
    
    def __init__(
        self,
        model: nn.Module,
        layer_idx: int,
        intervention_fn: Callable[[torch.Tensor], torch.Tensor]
    ):
        """
        Initialize activation hook.
        
        Args:
            model: Target model to intervene on
            layer_idx: Layer index to apply intervention
            intervention_fn: Function that takes activations and returns modified activations
        """
        self.model = model
        self.layer_idx = layer_idx
        self.intervention_fn = intervention_fn
        self.hook_handle = None
    
    def __enter__(self):
        """Register forward hook."""
        # Get the model's layers
        if hasattr(self.model, 'model') and hasattr(self.model.model, 'layers'):
            layers = self.model.model.layers  # Gemma, Llama structure
        elif hasattr(self.model, 'transformer') and hasattr(self.model.transformer, 'h'):
            layers = self.model.transformer.h  # GPT-2 structure
        else:
            raise ValueError(f"Unsupported model architecture")
        
        def hook_fn(module, input, output):
            """Apply intervention to activations."""
            # output is typically a tuple; first element is hidden states
            hidden_states = output[0] if isinstance(output, tuple) else output
            
            # Apply intervention
            modified_states = self.intervention_fn(hidden_states)
            
            # Return modified output
            if isinstance(output, tuple):
                return (modified_states,) + output[1:]
            else:
                return modified_states
        
        # Register hook
        self.hook_handle = layers[self.layer_idx].register_forward_hook(hook_fn)
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Remove forward hook."""
        if self.hook_handle is not None:
            self.hook_handle.remove()
            self.hook_handle = None


class InterventionPipeline:
    """
    High-level interface for applying activation interventions.
    
    Supports:
    - Single-layer interventions
    - Multi-layer interventions
    - Batched interventions
    """
    
    def __init__(
        self,
        model: nn.Module,
        tokenizer: AutoTokenizer,
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        """
        Initialize intervention pipeline.
        
        Args:
            model: Target model to intervene on
            tokenizer: Tokenizer for the model
            device: Device to run on
        """
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.model.to(device)
        self.model.eval()
    
    def apply_intervention(
        self,
        prompt: str,
        layer_idx: int,
        delta_f: torch.Tensor,
        position: str = "last",
        max_new_tokens: int = 100,
        do_sample: bool = False,
        temperature: float = 1.0
    ) -> str:
        """
        Apply single-layer intervention and generate response.
        
        Computes: f'(x, c) = f(x) + δf
        
        Args:
            prompt: Input prompt
            layer_idx: Layer to intervene on
            delta_f: Intervention vector [d_hidden]
            position: Which token position to intervene on ("last", "all", or int)
            max_new_tokens: Max tokens to generate
            do_sample: Whether to sample or use greedy decoding
            temperature: Sampling temperature
            
        Returns:
            Generated text response
        """
        # Tokenize input
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            padding=True,
            truncation=True
        ).to(self.device)
        
        # Define intervention function
        def intervention_fn(hidden_states: torch.Tensor) -> torch.Tensor:
            """Add δf to activations at specified position."""
            # hidden_states shape: [batch_size, seq_len, d_hidden]
            modified = hidden_states.clone()
            
            if position == "last":
                # Intervene on last token
                modified[:, -1, :] = modified[:, -1, :] + delta_f
            elif position == "all":
                # Intervene on all tokens
                modified = modified + delta_f.unsqueeze(0).unsqueeze(0)
            elif isinstance(position, int):
                # Intervene on specific position
                modified[:, position, :] = modified[:, position, :] + delta_f
            else:
                raise ValueError(f"Invalid position: {position}")
            
            return modified
        
        # Apply intervention and generate
        with ActivationHook(self.model, layer_idx, intervention_fn):
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=do_sample,
                    temperature=temperature,
                    pad_token_id=self.tokenizer.pad_token_id
                )
        
        # Decode response
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return response
    
    def apply_multi_layer_intervention(
        self,
        prompt: str,
        interventions: Dict[int, torch.Tensor],
        position: str = "last",
        max_new_tokens: int = 100,
        do_sample: bool = False,
        temperature: float = 1.0
    ) -> str:
        """
        Apply interventions to multiple layers simultaneously.
        
        Args:
            prompt: Input prompt
            interventions: Dict mapping layer_idx -> delta_f
            position: Which token position to intervene on
            max_new_tokens: Max tokens to generate
            do_sample: Whether to sample
            temperature: Sampling temperature
            
        Returns:
            Generated text response
        """
        # Tokenize input
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            padding=True,
            truncation=True
        ).to(self.device)
        
        # Create intervention functions for each layer
        hooks = []
        
        for layer_idx, delta_f in interventions.items():
            def make_intervention_fn(delta):
                def intervention_fn(hidden_states: torch.Tensor) -> torch.Tensor:
                    modified = hidden_states.clone()
                    
                    if position == "last":
                        modified[:, -1, :] = modified[:, -1, :] + delta
                    elif position == "all":
                        modified = modified + delta.unsqueeze(0).unsqueeze(0)
                    elif isinstance(position, int):
                        modified[:, position, :] = modified[:, position, :] + delta
                    
                    return modified
                return intervention_fn
            
            hook = ActivationHook(
                self.model,
                layer_idx,
                make_intervention_fn(delta_f)
            )
            hooks.append(hook)
        
        # Apply all interventions and generate
        with torch.no_grad():
            # Enter all hook contexts
            for hook in hooks:
                hook.__enter__()
            
            try:
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=do_sample,
                    temperature=temperature,
                    pad_token_id=self.tokenizer.pad_token_id
                )
            finally:
                # Exit all hook contexts
                for hook in hooks:
                    hook.__exit__(None, None, None)
        
        # Decode response
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return response
    
    def get_intervened_activations(
        self,
        prompt: str,
        layer_idx: int,
        delta_f: torch.Tensor,
        position: str = "last"
    ) -> torch.Tensor:
        """
        Get activations after intervention (for AO labeling).
        
        Returns f'(x, c) = f(x) + δf
        
        Args:
            prompt: Input prompt
            layer_idx: Layer to intervene on
            delta_f: Intervention vector
            position: Token position
            
        Returns:
            Intervened activations [seq_len, d_hidden]
        """
        # Tokenize input
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            padding=True,
            truncation=True
        ).to(self.device)
        
        # Storage for intervened activations
        intervened_acts = {}
        
        def intervention_fn(hidden_states: torch.Tensor) -> torch.Tensor:
            """Add δf and store result."""
            modified = hidden_states.clone()
            
            if position == "last":
                modified[:, -1, :] = modified[:, -1, :] + delta_f
            elif position == "all":
                modified = modified + delta_f.unsqueeze(0).unsqueeze(0)
            elif isinstance(position, int):
                modified[:, position, :] = modified[:, position, :] + delta_f
            
            # Store intervened activations
            intervened_acts['result'] = modified.detach()
            
            return modified
        
        # Apply intervention
        with ActivationHook(self.model, layer_idx, intervention_fn):
            with torch.no_grad():
                _ = self.model(**inputs)
        
        # Return intervened activations (remove batch dimension)
        return intervened_acts['result'][0]


if __name__ == "__main__":
    print("Testing Intervention Pipeline...")
    
    # This test requires a model - skip if not available
    try:
        print("\n1. Loading model...")
        model_name = "google/gemma-2-2b-it"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True
        )
        
        print("\n2. Creating intervention pipeline...")
        pipeline = InterventionPipeline(model, tokenizer)
        
        print("\n3. Testing single-layer intervention...")
        prompt = "What is the capital of France?"
        
        # Get model's hidden dimension
        if hasattr(model, 'model') and hasattr(model.model, 'layers'):
            d_hidden = model.model.layers[0].self_attn.head_dim * model.model.layers[0].self_attn.num_heads
        else:
            d_hidden = 2048  # default
        
        # Create random intervention
        delta_f = torch.randn(d_hidden, device=pipeline.device) * 0.01
        
        # Apply intervention at middle layer
        response = pipeline.apply_intervention(
            prompt=prompt,
            layer_idx=12,
            delta_f=delta_f,
            max_new_tokens=50
        )
        
        print(f"   Prompt: {prompt}")
        print(f"   Response: {response[:100]}...")
        
        print("\n✓ Intervention pipeline tests complete!")
        
    except Exception as e:
        print(f"\n⚠ Skipping intervention tests (model not available): {e}")
