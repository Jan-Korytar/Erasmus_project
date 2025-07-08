import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

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
    def __init__(self, channel_in, channel_out, kernel_size):
        super().__init__()
        self.conv = nn.Conv2d(channel_in, channel_out, kernel_size=kernel_size, padding=1, )
        self.upsample = nn.Upsample(scale_factor=2, mode="bilinear")

    def forward(self, x):
        # x.shape = (B, C, H, W):
        x = self.upsample(x)
        x = F.relu(self.conv(x))
        return x

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

class BertModel(nn.Module):
    def __init__(self,):
        super().__init__()



class Decoder(nn.Module):
    def __init__(self, text_embed_dim, latent_size=(512, 8, 8), depth =3, output_size = (3, 215, 215)):
        super().__init__()
        self.output_size = output_size
        self.latent_size = latent_size
        self.text_embed_dim = text_embed_dim
        self.seq_len = 168 # TODO take from config
        self.latent = nn.Parameter(torch.randn(latent_size))
        self.depth = depth

        current_channels = latent_size[0]

        #self.f1 = nn.Linear(text_embed_dim * self.seq_len, latent_size[0] * latent_size[1] * latent_size[2])
        self.text_attention = CrossAttention(num_heads=4, embed_dim=current_channels, map_dim=text_embed_dim)


        self.upscale_layers = nn.ModuleList()
        self.cross_attention_layers = nn.ModuleList()


        for _ in range(depth):
            self.upscale_layers.append(
                ConvTranspose(channel_in=current_channels, channel_out=current_channels // 2, kernel_size=4))
            current_channels = current_channels // 2
            self.cross_attention_layers.append(CrossAttention(num_heads=4, embed_dim=current_channels, map_dim=text_embed_dim))
        self.last_conv = nn.Conv2d(current_channels, 3, kernel_size=1)

    def forward(self, encoder_output):
        batch_size = encoder_output.shape[0]
        #x = (F.relu(self.f1(encoder_output.view(batch_size, -1))).view(batch_size, *self.latent_size))
        latent = self.latent.repeat(batch_size, 1, 1, 1)
        x = self.text_attention(encoder_output, latent)
        for upscale_layer, attention_layer in zip(self.upscale_layers, self.cross_attention_layers):
            x = upscale_layer(x)
            x = attention_layer(encoder_output, x)
        x = self.last_conv(x)
        x = F.interpolate(x, size=self.output_size[1:], mode="bilinear", align_corners=True)
        return x




