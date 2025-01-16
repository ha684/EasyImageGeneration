import gradio as gr
import os
import torch
from PIL import Image
from model_load import load_llama_model, load_stablediffusion_model
from utils import generate_image, generate_image_prompt

# Sample topics with descriptions
SAMPLE_TOPICS = {
    "Cyberpunk City": "A futuristic cityscape with neon lights and flying vehicles",
    "Enchanted Forest": "A magical forest with glowing flowers and mystical creatures",
    "Space Station": "An orbital space station with Earth visible in the background",
    "Underwater Temple": "Ancient ruins submerged in a crystal-clear ocean",
    "Desert Oasis": "A lush oasis surrounded by golden sand dunes",
    "Steampunk Workshop": "A Victorian-era workshop filled with brass machinery and steam engines",
    "Crystal Cave": "A cave filled with giant, colorful crystals and mysterious light",
    "Floating Islands": "Magical islands floating in the sky with waterfalls"
}

def extract_topic_from_url(url):
    """
    Placeholder function to extract topic from a paper URL
    You can implement paper parsing logic here
    """
    # TODO: Implement paper parsing
    return "Sample topic from paper"

def process_input(input_text, input_type="text", history=[]):
    """
    Process either direct topic input or URL
    """
    if input_type == "url":
        topic = extract_topic_from_url(input_text)
    else:
        topic = input_text
        
    # Generate prompt from topic
    prompt = generate_image_prompt(llama_pipe, topic)
    
    # Generate image from prompt
    image = generate_image(sd_pipe, prompt)
    
    # Update chat history
    history.append((input_text, f"Generated prompt: {prompt}"))
    
    return image, history

def clear_conversation():
    """
    Clear the conversation history
    """
    return [], None

def use_sample_topic(topic):
    """
    Handle sample topic selection
    """
    return topic

# Load models at startup
print("Loading models...")
llama_pipe = load_llama_model()
sd_pipe = load_stablediffusion_model()
print("Models loaded successfully!")

# Create Gradio interface
with gr.Blocks() as demo:
    gr.Markdown("# AI Image Generation from Topics or Papers")
    
    with gr.Row():
        # Image display area
        output_image = gr.Image(type="pil", label="Generated Image")
    
    with gr.Row():
        # Input type selector
        input_type = gr.Radio(
            choices=["text", "url"],
            value="text",
            label="Input Type",
            info="Choose whether to input a topic directly or provide a paper URL"
        )
    
    with gr.Row():
        # Text input area
        input_text = gr.Textbox(
            placeholder="Enter your topic or paper URL here...",
            label="Input",
            lines=2
        )
    
    # Sample topics gallery
    with gr.Row():
        gr.Markdown("### Try These Sample Topics")
    
    with gr.Row():
        sample_gallery = gr.Gallery(
            label="Sample Topics",
            value=[
                (None, topic, description) 
                for topic, description in SAMPLE_TOPICS.items()
            ],
            columns=4,
            object_fit="contain",
            height="auto"
        ).style(grid=2)
    
    with gr.Row():
        # Buttons
        submit_btn = gr.Button("Generate", variant="primary")
        clear_btn = gr.Button("Clear")
    
    # Chat history area
    chatbot = gr.Chatbot(
        label="Generation History",
        height=300
    )
    
    # Handle button clicks and gallery selection
    submit_btn.click(
        fn=process_input,
        inputs=[input_text, input_type, chatbot],
        outputs=[output_image, chatbot]
    )
    
    clear_btn.click(
        fn=clear_conversation,
        inputs=[],
        outputs=[chatbot, output_image]
    )
    
    # Handle sample gallery clicks
    sample_gallery.select(
        fn=use_sample_topic,
        inputs=[],
        outputs=input_text
    )
    
    # Update placeholder based on input type
    def update_placeholder(input_type):
        if input_type == "url":
            return gr.Textbox.update(placeholder="Enter paper URL here...")
        return gr.Textbox.update(placeholder="Enter your topic here...")
    
    input_type.change(
        fn=update_placeholder,
        inputs=[input_type],
        outputs=[input_text]
    )

# Launch the app
if __name__ == "__main__":
    demo.launch(share=True)