import os
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from diffusers import StableDiffusion3Pipeline,BitsAndBytesConfig, SD3Transformer2DModel

def load_llama_model(model_id="meta-llama/Llama-3.2-3B-Instruct"):
    model = AutoModelForCausalLM.from_pretrained(
        model_id, 
        device_map="cuda:1", 
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
    nf4_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16
    )
    model_nf4 = SD3Transformer2DModel.from_pretrained(
        model_id,
        subfolder="transformer",
        quantization_config=nf4_config,
        torch_dtype=torch.float16
    )

    pipeline = StableDiffusion3Pipeline.from_pretrained(
        model_id, 
        transformer=model_nf4,
        torch_dtype=torch.float16
    )
    pipeline.enable_model_cpu_offload()
    return pipeline