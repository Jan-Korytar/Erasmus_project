import random
from pathlib import Path

import torchvision
import torchvision.transforms as T
import yaml
from torch.utils.data import Dataset

from utils.helpers import get_project_root


class TextAndImageDataset(Dataset):
    """
    Dataset combining text and images with optional augmentations.

    Args:
        text_path (str or Path): Path to the text file containing text entries.
        image_path (str or Path): Directory containing image files (*.jpg).
        augment_images (bool): Whether to apply image augmentations. Default is False.
        augment_text (bool): Whether to apply text augmentations. Default is True.
    """

    def __init__(self, text_path, image_path, augment_images=False, augment_text=True):
        self.image_paths = sorted(Path(image_path).glob('*.jpg'), key=lambda p: p.name)
        self.augment_images = False
        self.augment_text = augment_text
        self.image_transform = T.Compose([
            T.RandomResizedCrop(128, scale=(0.9, 1.0)),
            T.RandomRotation(degrees=5),
        ]) if augment_images else None

        self.normalize = T.Normalize((0.5,), (0.5,))

        with open(get_project_root() / 'config.yml') as f:
            config = yaml.safe_load(f)
            self.images = config['training']['images']

        with open(text_path, 'r', encoding='utf-8') as f:
            self.text = f.read().split('\n')[:-1]




    def __len__(self):
        return len(self.text)

    def text_augmentation(self, text: str):
        if not self.augment_text:
            return '_', text

        org_name, text = text.split(';', 1)
        sentences = text.split('.')
        keep = random.uniform(0.2, 0.6)
        sentences = random.sample(sentences, max(1, int(keep * (len(sentences)))))

        def drop_words(sentence):
            tokens = sentence.split()
            kept = [w if random.random() > 0.10 else '[MASK]' for w in tokens]
            return ' '.join(kept)

        sentences = [drop_words(s) for s in sentences]
        if random.random() < 0.65:
            name = '[NAME]'
        else:
            name = org_name

        insert_pos = random.randint(0, int(len(sentences) * .66))  # should not be at the end
        sentences.insert(insert_pos, name)

        text = '. '.join(sentences)

        return org_name, text


    def __getitem__(self, idx):
        idx = idx % self.images  # for overfitting
        # Dataset Mean: [0.8937776  0.88624966 0.87821686]
        # Dataset Std: [0.20348613 0.20895252 0.21951194]
        image = torchvision.io.read_image(self.image_paths[idx]) / 255.0
        if self.augment_images:
            image = self.image_transform(image)
        image = self.normalize(image)

        name, text = self.text_augmentation(self.text[idx])

        return image, text, name
