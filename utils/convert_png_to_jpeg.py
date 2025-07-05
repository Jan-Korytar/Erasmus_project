import os
from pathlib import Path
from PIL import Image
from tqdm import tqdm
import numpy as np
import torch


def convert_images_to_jpg(input_dir, output_dir, size=(128, 128)):
    os.makedirs(output_dir, exist_ok=True)
    image_paths = list(Path(input_dir).glob('*.png'))

    means = []
    stds = []

    for image_path in tqdm(image_paths, total=len(image_paths)):
        with Image.open(image_path) as im:
            im = im.convert("RGBA")
            background = Image.new("RGBA", im.size, (255, 255, 255, 255))
            im = Image.alpha_composite(background, im).convert("RGB")
            im = im.resize(size, Image.LANCZOS)

            # Save as jpg
            out_name = os.path.basename(image_path)[:-3] + 'jpg'
            output_path = os.path.join(output_dir, out_name)
            im.save(output_path, format="JPEG", quality=100)

            # Convert to numpy array, normalize pixels to [0,1]
            img_np = np.array(im).astype(np.float32) / 255.0
            means.append(img_np.mean(axis=(0, 1)))  # per channel mean
            stds.append(img_np.std(axis=(0, 1)))  # per channel std

    # Calculate dataset mean and std over all images
    mean = np.mean(means, axis=0)
    std = np.mean(stds, axis=0)

    print(f"Dataset Mean: {mean}")
    print(f"Dataset Std: {std}")
    return mean, std


convert_images_to_jpg(r'C:\My folder\pokedex-main\images\small_images', r'C:\My folder\Erasmus_project\data\images')
