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
config_path = get_project_root() / 'config.yaml'
with open(config_path, 'r') as f:
    config = yaml.safe_load(f)

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

# Initialize decoder
decoder = Decoder(
    text_embed_dim=encoder.config.hidden_size,
    latent_size=config['model']['latent_size'],
    decoder_depth=config['model']['decoder_depth'],
    output_size=config['model']['output_size']
)

# Load weights
decoder_weights = get_project_root() / 'models' / 'decoder_weights.pth'
encoder_weights = get_project_root() / 'models' / 'encoder_weights.pth'

state_dict = torch.load(decoder_weights, map_location='cuda')
# Strip 'module.' prefix if present
new_state_dict_decoder = OrderedDict()
for k, v in state_dict.items():
    name = k
    if k.startswith('module.'):
        name = k[7:]  # remove 'module.' prefix
    new_state_dict_decoder[name] = v

decoder.load_state_dict(new_state_dict_decoder)
state_dict = torch.load(encoder_weights, map_location='cuda')
# Strip 'module.' prefix if present
new_state_dict_encoder = OrderedDict()
for k, v in state_dict.items():
    name = k
    if k.startswith('module.'):
        name = k[7:]  # remove 'module.' prefix
    new_state_dict_encoder[name] = v

encoder.load_state_dict(new_state_dict_encoder)

decoder.eval()
encoder.eval()


def generate_image(prompt):
    with torch.no_grad():
        tokens = tokenizer(prompt, return_tensors='pt', max_length=120, truncation=True, padding='max_length')
        text_embedding = encoder(**tokens).last_hidden_state
        image_tensor = decoder(text_embedding)[0]
        image_tensor = (image_tensor + 1) / 2
        image_tensor = torch.clamp(image_tensor, 0, 1).to('cpu')
        image_np = (image_tensor.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        return Image.fromarray(image_np)


demo = gr.Interface(
    fn=generate_image,
    inputs=gr.Textbox(lines=2, placeholder="Enter a prompt...", label="Text Prompt"),
    outputs=gr.Image(type="pil", label="Generated Image"),
    title="Text-to-Image Generator",
    description="Enter a prompt to generate an image using your trained decoder."
)

demo.launch(share=True)
