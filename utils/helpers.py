import os
import shutil
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from matplotlib import pyplot as plt
from torchvision.utils import save_image
from tqdm import tqdm


def get_project_root() -> Path:
    return Path(__file__).parent.parent

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


def plot_train_val_losses(train_losses, val_losses):
    epochs = range(1, len(train_losses) + 1)

    fig, ax1 = plt.subplots()

    # Primary y-axis for training loss
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Train Loss', color='tab:blue')
    ax1.plot(epochs, train_losses, label='Train Loss', color='tab:blue')
    ax1.tick_params(axis='y', labelcolor='tab:blue')

    # Secondary y-axis for validation loss
    ax2 = ax1.twinx()
    ax2.set_ylabel('Validation Loss', color='tab:red')
    ax2.plot(epochs, val_losses, label='Val Loss', color='tab:red')
    ax2.tick_params(axis='y', labelcolor='tab:red')

    plt.title('Training vs Validation Loss')
    fig.tight_layout()
    plt.grid(True)
    plt.savefig(get_project_root() / 'utils' / 'train_val_loss.jpg')
    plt.close()


def save_sample_images(tensor, filename, unnormalize=True):
    """
    Saves a batch of images to a file.
    tensor: (B, 3, H, W)
    """

    path = get_project_root() / 'utils' / 'outputs'
    if os.path.exists(path) and filename == '0_0.jpg':
        shutil.rmtree(path)
    os.makedirs(path, exist_ok=True)

    if unnormalize:
        # mean = torch.tensor([0.8937776, 0.88624966, 0.87821686]).view(1, 3, 1, 1).to(device)
        # std = torch.tensor([0.20348613, 0.20895252, 0.21951194]).view(1, 3, 1, 1).to(device)
        tensor = (tensor + 1) / 2
        tensor = torch.clamp(tensor, 0, 1)

    save_image(tensor, path / filename, nrow=4)


if __name__ == "__main__":
    print(get_project_root())
    convert_images_to_jpg(r'C:\My folder\pokedex-main\images\small_images',
                          r'C:\My folder\Erasmus_project\data\images\215', size=(215, 215))
