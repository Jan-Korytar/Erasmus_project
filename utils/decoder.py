import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from helpers import get_project_root
import yaml

#latent space 256, 8, 8

class CrossAttention(nn.Module):
    def __init__(self, num_heads=8, embed_dim=256, map_dim=None):
        super().__init__()
        if map_dim is None:
            map_dim = embed_dim
        self.attention = nn.MultiheadAttention(num_heads=num_heads, embed_dim=embed_dim,vdim=map_dim, kdim=map_dim, batch_first=True, dropout=0.1)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, encoder_output, decoder_input_latent):
        B, C, H, W = decoder_input_latent.shape
        decoder_input_latent = decoder_input_latent.view(B, C, H * W).permute(0, 2, 1) # -> B, HW, C
        attn_output, attn_output_weights = self.attention(query=decoder_input_latent, key=encoder_output, value=encoder_output)
        output = self.norm(decoder_input_latent + attn_output)
        output = output.permute(0, 2, 1).view(B, C, H, W)
        return output

class Upscale(nn.Module):
    def __init__(self, channel_in, channel_out, kernel_size, use_conv2d = False):
        super().__init__()
        self.conv = nn.Conv2d(channel_in, channel_out, kernel_size=kernel_size, padding=1, ) if use_conv2d else nn.Identity()
        self.upsample = nn.Upsample(scale_factor=2, mode="bilinear")

    def forward(self, x):
        # x.shape = (B, C, H, W):
        x = self.upsample(x)
        x = F.relu(self.conv(x))
        return x

class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.GroupNorm(8, in_channels),
            nn.GELU(),
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.GroupNorm(8, out_channels),
            nn.GELU(),
            nn.Conv2d(out_channels, out_channels, 3, padding=1)
        )
        self.skip = nn.Conv2d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()

    def forward(self, x):
        return self.block(x) + self.skip(x)

class ConvTranspose(nn.Module):
    def __init__(self, channel_in, channel_out, kernel_size):
        super().__init__()
        self.deconv = nn.ConvTranspose2d(
            in_channels=channel_in,
            out_channels=channel_out,
            kernel_size=kernel_size,
            stride=2,
            padding=1,
            output_padding=1
        )

    def forward(self, x):
        # x.shape = (B, C, H, W)
        x = F.relu(self.deconv(x))
        return x



class Decoder(nn.Module):
    def __init__(self, text_embed_dim, latent_size=(512, 8, 8), num_heads=8, decoder_depth =3, output_size = (3, 215, 215), seq_len=168 ):
        super().__init__()
        self.output_size = output_size
        self.latent_size = latent_size
        self.text_embed_dim = text_embed_dim
        self.seq_len = seq_len
        self.latent = nn.Parameter(torch.randn(latent_size))
        self.depth = decoder_depth

        current_channels = latent_size[0]

        #self.f1 = nn.Linear(text_embed_dim * self.seq_len, latent_size[0] * latent_size[1] * latent_size[2])
        self.text_attention = CrossAttention(num_heads=num_heads, embed_dim=current_channels, map_dim=text_embed_dim)
        self.upscale = Upscale(None, None, None, False)

        self.resblock_layers = nn.ModuleList()
        self.cross_attention_layers = nn.ModuleList()


        for _ in range(decoder_depth):
            self.resblock_layers.append(
                ResBlock(in_channels=current_channels, out_channels=current_channels//2))
            current_channels = current_channels // 2
            self.cross_attention_layers.append(CrossAttention(num_heads=num_heads, embed_dim=current_channels, map_dim=text_embed_dim))
        self.last_conv = nn.Conv2d(current_channels, 3, kernel_size=1)

    def forward(self, encoder_output):
        batch_size = encoder_output.shape[0]
        #x = (F.relu(self.f1(encoder_output.view(batch_size, -1))).view(batch_size, *self.latent_size))
        latent = self.latent.repeat(batch_size, 1, 1, 1)
        x = self.text_attention(encoder_output, latent)
        for resblock_layer, attention_layer in zip(self.resblock_layers, self.cross_attention_layers):
            x = resblock_layer(x)
            x = self.upscale(x)
            x = attention_layer(encoder_output, x)
        x = F.tanh(self.last_conv(x))
        x = F.interpolate(x, size=self.output_size[1:], mode="bilinear", align_corners=True)
        return x



if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    project_root = get_project_root()
    with open(project_root / 'config.yaml') as f:
        config = yaml.safe_load(f)
        training_config = config['training']
        model_config = config['model']
    decoder = Decoder(text_embed_dim=model_config['text_embed_dim'], latent_size=model_config['latent_size'],
                      decoder_depth=model_config['decoder_depth'], output_size=model_config['output_size']).to(device)

    out = decoder(torch.rand(2, 168, 256).to(device))

