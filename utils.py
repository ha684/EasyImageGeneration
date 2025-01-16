def generate_image_prompt(pipe, topic):
    
    generation_args = {
        "max_new_tokens": 100,
        "return_full_text": False,
        "temperature": 1.5,
        "do_sample": True,
    }
    messages = [
        {"role": "system", "content": "You are a helpfull assistant"},
        {"role": "user", "content": "You are an expert in creating optimized prompts for Stable Diffusion image generation. "
                                "Given a topic or concept, you create detailed, descriptive prompts that help Stable Diffusion "
                                "generate the best possible images. Your prompts should include:\n\n"
                                "- Detailed visual descriptions (colors, lighting, composition, perspective)\n"
                                "- Art style specifications (photorealistic, anime, oil painting, etc.)\n"
                                "- Technical parameters (quality boosters like 'high resolution', 'detailed', 'masterpiece')\n"
                                "- Atmosphere and mood descriptors\n\n"
                                "Format your responses as complete, ready-to-use prompts that capture the essence of the topic "
                                "while providing enough detail for high-quality image generation. Avoid negative prompts or "
                                f"technical parameters like steps or guidance scale. Create an optimized Stable Diffusion prompt for this topic: {topic}"}
    ]
    output = pipe(messages, **generation_args)
    
    return str(output[0]['generated_text'])

def generate_image(pipe,prompt):
    image = pipe(
        prompt,
        num_inference_steps=40,
        guidance_scale=4.5,
    ).images[0]
    return image