from pathlib import Path

import torch
import torchvision
import torchvision.transforms as T
from torch.utils.data import Dataset
from transformers import BertTokenizer, BertModel


class TextAndImageDataset(Dataset):
    def __init__(self, text_path, image_path, augment_images=True, pad_sentences=False, return_hidden=True):
        self.image_paths = list(Path(image_path).glob('*.jpg'))
        self.pad_sentences = pad_sentences
        self.augment_images = augment_images
        self.image_transform = T.Compose([
            T.RandomResizedCrop(128, scale=(0.9, 1.0)),
            T.RandomHorizontalFlip(),
            T.RandomRotation(degrees=5),
        ]) if augment_images else T.Resize((128, 128))
        self.tokenizer = BertTokenizer.from_pretrained("prajjwal1/bert-mini",)
        self.return_hidden = return_hidden
        self.normalize = T.Normalize((0.5,), (0.5,))
        if return_hidden:
            self.model = BertModel.from_pretrained("prajjwal1/bert-mini")

        with open(text_path, 'r', encoding='utf-8') as f:
            self.text = f.read().split('\n')[1:-1]




    def __len__(self):
        return len(self.text)

    def __getitem__(self, idx):
        # Dataset Mean: [0.8937776  0.88624966 0.87821686]
        # Dataset Std: [0.20348613 0.20895252 0.21951194]
        image = torchvision.io.read_image(self.image_paths[idx]) / 255.0
        if self.augment_images:
            image = self.image_transform(image)
        image = self.normalize(image)
        text =  self.text[idx]
        if self.pad_sentences:
            embed = self.tokenizer(text,return_tensors="pt", truncation=True, max_length=168, return_attention_mask=True, padding='max_length')
        else:
            embed = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=168, return_attention_mask=True)



        if self.return_hidden:
            embed = self.model(**embed)
            return image, embed.last_hidden_state.squeeze()
        else:
            for k in embed:
                embed[k] = embed[k].squeeze()

            return image, embed



