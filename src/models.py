"""
Module 2: Base Models - Load Karvonen Activation Oracle
"""
from transformers import AutoTokenizer, AutoModelForCausalLM

def load_activation_oracle(model_name="adamkarvonen/gemma-2-2b-activation-oracle"):
    """Load pretrained activation oracle from HF"""
    # TODO: Real loading when GPU available
    tokenizer = None
    model = None
    print(" Module 2 stub: Karvonen AO ready (GPU needed)")
    return tokenizer, model

if __name__ == "__main__":
    tokenizer, model = load_activation_oracle()
    print("Module 2: Models ready")
