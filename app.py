import gradio as gr
import os
import torch
from PIL import Image
from model_load import load_llama_model, load_stablediffusion_model
from utils import generate_image_prompt, generate_image
import requests
from bs4 import BeautifulSoup
import re

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

def is_url(text):
    url_pattern = re.compile(
        r'^https?://'  # http:// or https://
        r'(?:(?:[A-Z0-9](?:[A-Z0-9-]{0,61}[A-Z0-9])?\.)+[A-Z]{2,6}\.?|'  # domain...
        r'localhost|'  # localhost...
        r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})'  # ...or ip
        r'(?::\d+)?'  # optional port
        r'(?:/?|[/?]\S+)$', re.IGNORECASE)
    return bool(url_pattern.match(text))

def extract_topic_from_url(url):
    response = requests.get(url)
    soup = BeautifulSoup(response.content, 'html.parser')
    news_box_div = soup.find("div", {"class": "list__news-up cate"})
    news_items = news_box_div.find_all("div", {"class":"box-category-content"})
    news_array = []
    for item in news_items[:1]:
        title = item.find("a").text.strip()
        intro = item.find("p").text.strip()
        news_array.append(title + "\n" + intro)
    return news_array[0]

def process_input(input_text, history=[]):
    """
    Process input text, automatically detecting if it's a URL or topic
    """
    if not input_text:
        return None, None, history
        
    # Determine if input is URL or topic
    if is_url(input_text):
        try:
            topic = extract_topic_from_url(input_text)
        except Exception as e:
            return None, f"Error processing URL: {str(e)}", history
    else:
        topic = input_text
        
    # Generate prompt from topic
    prompt = generate_image_prompt(llama_pipe, topic)
    
    # Generate image from prompt with progress bar
    try:
        image = generate_image(sd_pipe, prompt)
    except Exception as e:
        return None, f"Error generating image: {str(e)}", history
    
    # Update chat history
    history.append((input_text, f"Generated prompt: {prompt}"))
    
    return image, prompt, history

def clear_conversation():
    """
    Clear the conversation history
    """
    return [], None, None

def use_sample_topic(evt: gr.SelectData):
    """
    Handle sample topic selection
    """
    topics = list(SAMPLE_TOPICS.keys())
    selected_topic = topics[evt.index]
    return selected_topic

# Load models at startup
print("Loading models...")
sd_pipe = load_stablediffusion_model()
llama_pipe = load_llama_model("microsoft/Phi-3.5-mini-instruct")
print("Models loaded successfully!")

# Create Gradio interface
with gr.Blocks() as demo:
    gr.Markdown("# AI Image Generation from Topics or URLs")
    
    with gr.Row():
        # Image display area
        output_image = gr.Image(type="pil", label="Generated Image")
    
    with gr.Row():
        # Text input area with automatic URL detection
        input_text = gr.Textbox(
            placeholder="Enter your topic or paste a URL here...",
            label="Input",
            lines=2
        )
    
    with gr.Row():
        # Generated prompt display
        generated_prompt = gr.Textbox(
            label="Generated Prompt",
            lines=2,
            interactive=False
        )
    
    # Sample topics gallery
    with gr.Row():
        gr.Markdown("### Try These Sample Topics")
    
    with gr.Row():
        # Create a grid of sample topics
        with gr.Column(scale=1):
            sample_grid = gr.Dataset(
                components=[gr.Textbox()],
                samples=[[topic] for topic in SAMPLE_TOPICS.keys()],
                label="Sample Topics",
                samples_per_page=8,
                type="index"
            )
    
    with gr.Row():
        # Buttons
        submit_btn = gr.Button("Generate", variant="primary")
        clear_btn = gr.Button("Clear")
    
    # Chat history area
    chatbot = gr.Chatbot(
        label="Generation History",
        height=300
    )
    
    # Handle button clicks and sample selection
    submit_btn.click(
        fn=process_input,
        inputs=[input_text, chatbot],
        outputs=[output_image, generated_prompt, chatbot]
    )
    
    clear_btn.click(
        fn=clear_conversation,
        inputs=[],
        outputs=[chatbot, output_image, generated_prompt]
    )
    
    # Handle sample selection
    sample_grid.select(
        fn=use_sample_topic,
        outputs=input_text
    )

# Launch the app
if __name__ == "__main__":
    demo.launch(share=True)