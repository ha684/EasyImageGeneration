import os
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from diffusers import StableDiffusion3Pipeline

def load_llama_model(model_id="meta-llama/Llama-3.2-3B-Instruct"):
    model = AutoModelForCausalLM.from_pretrained(
        model_id, 
        device_map="auto", 
        torch_dtype=torch.float16, 
        trust_remote_code=True, 
    )
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
    )
    
    return pipe

def load_stablediffusion_model(model_id="stabilityai/stable-diffusion-3.5-medium"):
    pipe = StableDiffusion3Pipeline.from_pretrained(
        model_id, 
        torch_dtype=torch.float16,
        device_map="balanced"
    )
    return pipe