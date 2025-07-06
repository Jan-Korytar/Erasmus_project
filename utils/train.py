import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from parso.python.tree import Class
from torch.utils.data import Dataset, DataLoader
from dataset import TextAndImageDataset
from tqdm import tqdm
from decoder import Decoder
from helpers import save_sample_images
'''device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

encoded = torch.rand(4, 168, 256).to(device)

decoder =Decoder(text_embed_dim=256, depth=3)

decoder.to(device)

output = decoder(encoded)'''




def train_decoder(decoder, dataloader, num_epochs=30, lr=1e-4, device='cuda'):
    decoder = decoder.to(device)
    optimizer = torch.optim.Adam(decoder.parameters(), lr=lr)
    criterion = nn.MSELoss()

    decoder.train()
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        for i, (target_image, text_embed) in enumerate(tqdm(dataloader)):
            text_embed = text_embed.to(device)            # shape: (B, seq_len, 256)
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

        # Optional: save image every few epochs


    return decoder




dataset = TextAndImageDataset('../data/text_description.csv', '../data/images', pad_sentences=True )
dataloader = DataLoader(dataset, batch_size=8, shuffle=True)
decoder = Decoder(text_embed_dim=256, depth=4, output_size=(3, 128, 128))

device = 'cuda' if torch.cuda.is_available() else 'cpu'
train_decoder(decoder, dataloader, num_epochs=40, device=device)




