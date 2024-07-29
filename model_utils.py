import torch
import os
import transformers
from tokenizers import AddedToken
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

def load_model(model_name_or_path, device="cuda"):
    print("Loading tokenizer and model from:", model_name_or_path)
    
    # load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_name_or_path,
        use_fast=True, # fast load tokenizer
        padding_side='right' # custom for rotary position embedding
    )

    # load model
    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        torch_dtype=torch.bfloat16,
        device_map=device,
        attn_implementation="flash_attention_2" # enable flash attention
    )

    # setting pad token by eos token if needed
    # model.generation_config.pad_token_id = model.generation_config.eos_token_id
    
    return model, tokenizer

def load_quantized_model(model_name_or_path, device="cuda"):
    print("Loading tokenizer and model with quantization config from:", model_name_or_path)
    # load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_name_or_path,
        use_fast=True, # fast load tokenizer
        padding_side='right' # custom for rotary position embedding
    )

    # BitsAndBytes config
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )

    # load model
    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        quantization_config=bnb_config,
        torch_dtype=torch.bfloat16,
        device_map=device,
        attn_implementation="flash_attention_2" # enable flash attention
    )

    return model, tokenizer
