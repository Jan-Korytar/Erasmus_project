from collections import OrderedDict

import gradio as gr
import numpy as np
import torch
import yaml
from PIL import Image
from transformers import AutoTokenizer, AutoModel

from utils.decoder import Decoder
from utils.helpers import get_project_root

# Load config
config_path = get_project_root() / 'config.yml'
with open(config_path, 'r') as f:
    config = yaml.safe_load(f)

device = 'cpu'

config['model']['num_heads'] = 16
config['model']['decoder_depth'] = 5
config['model']['latent_size'] = [1024, 4, 4]
config['model']['output_size'] = [3, 128, 128]
config['model']["bert_model"] = "distilbert/distilbert-base-uncased"

# Load tokenizer and encoder
tokenizer = AutoTokenizer.from_pretrained(config['model']["bert_model"])
tokenizer.add_special_tokens({'additional_special_tokens': ['[NAME]']})
encoder = AutoModel.from_pretrained(config['model']["bert_model"])
encoder.resize_token_embeddings(len(tokenizer))


decoder = Decoder(
    text_embed_dim=encoder.config.hidden_size,
    latent_size=config['model']['latent_size'],
    decoder_depth=config['model']['decoder_depth'],
    output_size=config['model']['output_size']
)


decoder_weights = get_project_root() / 'models' / 'decoder_weights.pth'
encoder_weights = get_project_root() / 'models' / 'encoder_weights.pth'

state_dict = torch.load(decoder_weights, map_location=device)
# Strip 'module.' prefix if present, beacuse of data parallelism
new_state_dict_decoder = OrderedDict()
for k, v in state_dict.items():
    name = k
    if k.startswith('module.'):
        name = k[7:]  # remove 'module.' prefix
    new_state_dict_decoder[name] = v

decoder.load_state_dict(new_state_dict_decoder)
state_dict = torch.load(encoder_weights, map_location=device)
new_state_dict_encoder = OrderedDict()
for k, v in state_dict.items():
    name = k
    if k.startswith('module.'):
        name = k[7:]  # remove 'module.' prefix
    new_state_dict_encoder[name] = v

encoder.load_state_dict(new_state_dict_encoder)

decoder.eval()
encoder.eval()


def load_prompts(file_path):
    prompts = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line or ';' not in line:
                continue
            name, desc = line.split(';', 1)
            desc = desc.strip()
            if desc:
                prompts.append(f"{name.strip()} - {desc}")
    return prompts


prompt_file = get_project_root() / 'data' / 'text_description_concat.csv'
prompt_options = load_prompts(prompt_file)


def generate_image_source(choice, dropdown_prompt, custom_prompt):
    prompt_text = dropdown_prompt if choice == "Predefined" else custom_prompt
    with torch.no_grad():
        tokens = tokenizer(prompt_text, return_tensors='pt', max_length=120, truncation=True, padding='max_length')
        text_embedding = encoder(**tokens).last_hidden_state
        image_tensor = decoder(text_embedding)[0]

        image_tensor = (image_tensor + 1) / 2
        image_tensor = torch.clamp(image_tensor, 0, 1).to(device)
        image_np = (image_tensor.permute(1, 2, 0).numpy() * 255).astype(np.uint8)

        image_128 = Image.fromarray(image_np)
        image_215 = image_128.resize((215, 215), Image.BICUBIC)

        return image_128, image_215


with gr.Blocks() as demo:
    gr.Markdown("### Text-to-Image Generator of pokemons")
    gr.Markdown(
        "There are two possibilities, one is to generate based on the training data. This also servers as information on training data. "
        "Ore one can choose to generate on custom input. For good results make the description long, similar to the training data and try with and without name"
    )

    choice = gr.Radio(["Predefined", "Custom"], label="Prompt Source", value="Predefined")

    with gr.Row():
        dropdown_prompt = gr.Dropdown(prompt_options, label="Predefined Prompts")
        custom_prompt = gr.Textbox(lines=2, label="Custom Prompt")

    generate_button = gr.Button("Generate Image")

    image_128 = gr.Image(type="pil", label="128x128 Image")
    image_215 = gr.Image(type="pil", label="215x215 Image")

    generate_button.click(
        generate_image_source,
        inputs=[choice, dropdown_prompt, custom_prompt],
        outputs=[image_128, image_215]
    )

demo.launch()
