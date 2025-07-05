import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from parso.python.tree import Class
import torchvision
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from PIL import Image

from transformers import BertTokenizer, BertModel



class TextAndImageDataset(Dataset):
    def __init__(self, text_path, image_path, transform=None, pad_sentences=False):
        self.image_paths = list(Path(image_path).glob('*.jpg'))
        self.pad_sentences = pad_sentences
        self.transform = transform
        self.tokenizer = BertTokenizer.from_pretrained("prajjwal1/bert-mini",)
        self.model = BertModel.from_pretrained("prajjwal1/bert-mini")
        with open(text_path, 'r', encoding='utf-8') as f:
            self.text = f.read().split('\n')[1:]




    def __len__(self):
        return len(self.text)

    def __getitem__(self, idx):
        # Dataset Mean: [0.8937776  0.88624966 0.87821686]
        # Dataset Std: [0.20348613 0.20895252 0.21951194]
        image = torchvision.io.read_image(self.image_paths[idx]) / 255.0
        if self.transform:
            image = self.transform(image)
        image -= torch.tensor([0.8937776,  0.88624966, 0.87821686])[:, None, None]
        image /= torch.tensor([0.20348613, 0.20895252, 0.21951194])[:, None, None]
        text =  self.text[idx]
        if self.pad_sentences:
            embed = self.tokenizer(text,return_tensors="pt", truncation=True, max_length=168, return_attention_mask=True, padding='max_length')
        else:
            embed = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=168, return_attention_mask=True)
        embed = self.model(**embed)

        return image, embed.last_hidden_state

dataset = TextAndImageDataset('../data/text_description.csv', '../data/images')
print(dataset[0])
