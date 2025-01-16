def generate_image_prompt(pipe, topic):
    messages = [
        {
            "role": "system",
            "content": """You are an expert in creating optimized prompts for Stable Diffusion image generation. 
            Given a topic or concept, you create detailed, descriptive prompts that help Stable Diffusion 
            generate the best possible images. Your prompts should include:
            
            - Detailed visual descriptions (colors, lighting, composition, perspective)
            - Art style specifications (photorealistic, anime, oil painting, etc.)
            - Technical parameters (quality boosters like 'high resolution', 'detailed', 'masterpiece')
            - Atmosphere and mood descriptors
            
            Format your responses as complete, ready-to-use prompts that capture the essence of the topic 
            while providing enough detail for high-quality image generation. Avoid negative prompts or 
            technical parameters like steps or guidance scale."""
        },
        {"role": "user", "content": f"Create an optimized Stable Diffusion prompt for this topic: {topic}"},
    ]
    
    outputs = pipe(
        messages,
    )
    
    return outputs


def generate_image(pipe,prompt):
    image = pipe(
        prompt,
        num_inference_steps=40,
        guidance_scale=4.5,
    ).images[0]
    return image