import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
from transformers import BertModel

from dataset import TextAndImageDataset
from decoder import Decoder
from helpers import save_sample_images, get_project_root
from perceptual_loss import PerceptualLoss


def validate(decoder, encoder, dataloader, device):
    decoder.eval()
    encoder.eval()
    criterion = nn.MSELoss(reduction='mean')
    val_loss = 0
    with torch.no_grad():
        for target_image, text_embed in dataloader:
            last_hidden = encoder(**text_embed.to(device)).last_hidden_state
            target_image = target_image.to(device)
            output = decoder(last_hidden)  # predicted image
            loss = criterion(output, target_image)
            val_loss += loss.item()

    val_loss /= len(dataloader)
    return val_loss


def train_decoder(decoder, encoder, train_dataloader, val_dataloader, num_epochs=30, lr=5e-3, device='cuda',
                  percpetual_loss=False):
    print(f'--- Beginning training with {device} ---')
    decoder = decoder.to(device)
    encoder = encoder.to(device)
    optimizer = torch.optim.Adam(decoder.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, num_epochs)
    mse_loss = nn.MSELoss(reduction='mean')
    perceptual_loss = PerceptualLoss().to(device) if percpetual_loss else None


    decoder.train()
    encoder.train()
    best_loss = float('inf')
    for epoch in range(num_epochs):
        epoch_train_loss = 0.0
        for i, (target_image, text_embed) in enumerate(tqdm(train_dataloader)):
            last_hidden = encoder(**text_embed.to(device)).last_hidden_state  # shape: (B, seq_len, 256)
            target_image: torch.Tensor = target_image.to(device)  # shape: (B, 3, H, W)

            optimizer.zero_grad()
            output = decoder(last_hidden)  # predicted image
            loss = (mse_loss(output, target_image) + (
                0.1 * perceptual_loss(output, target_image) if perceptual_loss else 0)
                    + (1e-3 * torch.mean(decoder.latent ** 2)))

            loss.backward()
            optimizer.step()

            epoch_train_loss += loss.item()
            if (i + 1) % 50 == 0:  # save image
                save_sample_images(output[:8], save_path=get_project_root() / 'utils' / 'outputs' / f'{epoch}_{i}.jpg')

        scheduler.step()
        val_loss = validate(decoder, encoder, val_dataloader, device)
        print(
            f"[Epoch {epoch + 1}/{num_epochs}] Loss: {epoch_train_loss / len(train_dataloader):.4f}, Val Loss: {val_loss}, LR: {scheduler.get_last_lr()[0]:.4f}")
        if val_loss < best_loss and epoch > 10:
            best_loss = val_loss
            print(f'Saving the best model at epoch {epoch + 1}/{num_epochs}')
            torch.save(decoder.state_dict(), "model_weights.pth")







if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    project_root = get_project_root()
    with open(project_root / 'config.yaml') as f:
        config = yaml.safe_load(f)
        training_config = config['training']
        model_config = config['model']
    full_dataset = TextAndImageDataset(project_root / 'data/text_description.csv', project_root / 'data/images',
                                       return_hidden=False)

    # Split datasets
    total_len = len(full_dataset)
    train_len = int(0.8 * total_len)
    val_len = int(0.1 * total_len)
    test_len = len(full_dataset) - train_len - val_len

    train_dataset, val_dataset, test_dataset = random_split(
        full_dataset,
        lengths=[train_len, val_len, test_len],
        generator=torch.Generator().manual_seed(42)  # for reproducibility
    )
    train_dataloader = DataLoader(train_dataset, batch_size=training_config['batch_size'], shuffle=True,
                                  pin_memory=True, num_workers=1, pin_memory_device=device)
    val_dataloader = DataLoader(val_dataset, batch_size=4, shuffle=False, pin_memory=True, num_workers=1,
                                pin_memory_device=device)
    decoder = Decoder(text_embed_dim=model_config['text_embed_dim'], latent_size=model_config['latent_size'],
                      decoder_depth=model_config['decoder_depth'], output_size=model_config['output_size'])
    bert_encoder = BertModel.from_pretrained("prajjwal1/bert-mini")

    train_decoder(decoder=decoder, encoder=bert_encoder, train_dataloader=train_dataloader, percpetual_loss=True,
                  val_dataloader=val_dataloader, num_epochs=training_config['num_epochs'],
                  lr=training_config['learning_rate'],
                  device=device)
