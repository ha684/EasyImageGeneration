import os
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
import torch
from transformers import pipeline
from diffusers import StableDiffusion3Pipeline

def load_llama_model(model_id="meta-llama/Llama-3.2-3B-Instruct"):
    pipe = pipeline(
        "text-generation",
        model=model_id,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    
    return pipe

def load_stablediffusion_model(model_id="stabilityai/stable-diffusion-3.5-medium"):
    pipe = StableDiffusion3Pipeline.from_pretrained(
        model_id, 
        torch_dtype=torch.float16,
    )
    return pipe