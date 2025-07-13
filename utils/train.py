import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from dataset import TextAndImageDataset
from tqdm import tqdm
from decoder import Decoder
from helpers import save_sample_images, get_project_root
from transformers import BertModel
from multiprocessing import freeze_support

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
    criterion = nn.MSELoss()

    decoder.train()
    encoder.train()
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        best_loss = float('inf')
        for i, (target_image, text_embed) in enumerate(tqdm(dataloader)):
            text_embed = encoder(**text_embed.to(device)).last_hidden_state            # shape: (B, seq_len, 256)
            target_image = target_image.to(device)        # shape: (B, 3, H, W)

            optimizer.zero_grad()
            output = decoder(text_embed)                  # predicted image
            loss = criterion(output, target_image)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            if (i + 1) % 20 == 0:
                save_sample_images(output[:8], save_path=f"outputs/{epoch}_{i + 1}.jpg")



        print(f"[Epoch {epoch+1}/{num_epochs}] Loss: {epoch_loss/len(dataloader):.4f}")
        if epoch_loss/len(dataloader) < best_loss and epoch > 10:
            best_loss = epoch_loss/len(dataloader)
            print(f'Saving the best model at epoch {epoch+1}/{num_epochs}')
            torch.save(decoder.state_dict(), "model_weights.pth")


        # Optional: save image every few epochs


    return decoder


if __name__ == '__main__':
    project_root = get_project_root()
    freeze_support()
    dataset = TextAndImageDataset(project_root / 'data/text_description.csv', project_root / 'data/images', pad_sentences=True, return_hiddens=False )
    dataloader = DataLoader(dataset, batch_size=8, shuffle=True, pin_memory=True, num_workers=4)
    decoder = Decoder(text_embed_dim=256, depth=4, output_size=(3, 128, 128))
    encoder = BertModel.from_pretrained("prajjwal1/bert-mini")

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    train_decoder(decoder=decoder,encoder=encoder , dataloader=dataloader, num_epochs=40, device=device)




