from multiprocessing import freeze_support

import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import BertModel

from dataset import TextAndImageDataset
from decoder import Decoder
from helpers import save_sample_images, get_project_root

'''device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

encoded = torch.rand(4, 168, 256).to(device)

decoder =Decoder(text_embed_dim=256, depth=3)

decoder.to(device)

output = decoder(encoded)'''


def validate(decoder, dataloader, device):
    decoder.eval()
    criterion = nn.MSELoss()
    val_loss = 0
    with torch.no_grad():
        for target_image, text_embed in dataloader:

            text_embed = text_embed.to(device)
            target_image = target_image.to(device)
            output = decoder(text_embed)                  # predicted image
            loss = criterion(output, target_image)
            val_loss += loss.item()

    val_loss /= len(dataloader.dataset)
    return val_loss




def train_decoder(decoder, encoder, dataloader, num_epochs=30, lr=1e-4, device='cuda'):
    print(f'Begining training with {device}')
    decoder = decoder.to(device)
    encoder = encoder.to(device)
    optimizer = torch.optim.Adam(decoder.parameters(), lr=lr)
    mseloss = nn.MSELoss()
    percieve_loss = 0


    decoder.train()
    encoder.train()
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        best_loss = float('inf')
        for i, (target_image, text_embed) in enumerate(tqdm(dataloader)):
            text_embed = encoder(**text_embed.to(device)).last_hidden_state            # shape: (B, seq_len, 256)
            target_image = target_image.to(device)      # shape: (B, 3, H, W)

            optimizer.zero_grad()
            output = decoder(text_embed)                  # predicted image
            loss = mseloss(output, target_image)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            if (i + 1) % 50 == 1:
                save_sample_images(output[:8], save_path=get_project_root() / 'utils' / 'outputs' / f'{epoch}_{i}.jpg')



        print(f"[Epoch {epoch+1}/{num_epochs}] Loss: {epoch_loss/len(dataloader):.4f}")
        if epoch_loss/len(dataloader) < best_loss and epoch > 10:
            pass
            best_loss = epoch_loss/len(dataloader)
            #print(f'Saving the best model at epoch {epoch+1}/{num_epochs}')
            #torch.save(decoder.state_dict(), "model_weights.pth")


        # Optional: save image every few epochs





if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    project_root = get_project_root()
    with open(project_root / 'config.yaml') as f:
        config = yaml.safe_load(f)
        training_config = config['training']
        model_config = config['model']
    freeze_support()
    dataset = TextAndImageDataset(project_root / 'data/text_description.csv', project_root / 'data/images', pad_sentences=True, return_hidden=False)
    dataloader = DataLoader(dataset, batch_size=training_config['batch_size'], shuffle=True, pin_memory=True, num_workers=2, pin_memory_device=device)
    decoder = Decoder(text_embed_dim=model_config['text_embed_dim'], latent_size=model_config['latent_size'],
                      decoder_depth=model_config['decoder_depth'], output_size=model_config['output_size'])
    bert_encoder = BertModel.from_pretrained("prajjwal1/bert-mini")

    train_decoder(decoder=decoder, encoder=bert_encoder, dataloader=dataloader, num_epochs=training_config['num_epochs'], device=device)




