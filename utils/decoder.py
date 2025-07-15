import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from torch.ao.nn.quantized import ConvTranspose2d
from torchinfo import summary

from helpers import get_project_root


#latent space 256, 8, 8

class CrossAttention(nn.Module):
    """
    Applies cross-attention between decoder latent and encoder output.
    """
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
    """
    Upsamples input feature maps using bilinear interpolation and optional convolution.
    """

    def __init__(self, in_channels, out_channels, use_conv2d=False):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, ) if use_conv2d else nn.Identity()
        self.upsample = nn.Upsample(scale_factor=2, mode="bilinear")

    def forward(self, x):
        # x.shape = (B, C, H, W):
        x = self.upsample(x)
        x = F.gelu(self.conv(x))
        return x

class ResBlock(nn.Module):
    """
    Residual block with two convolutional layers and skip connection.
    """
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


class ConvTranspose2d(nn.Module):
    """
    Applies transposed convolution (deconvolution) for upsampling.
    """

    def __init__(self, in_channels, out_channels, kernel_size=4):
        super().__init__()
        self.deconv = nn.ConvTranspose2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=2,
            padding=1,
            output_padding=0
        )

    def forward(self, x):
        # x.shape = (B, C, H, W)
        x = F.gelu(self.deconv(x))
        return x


class PixelShuffleUpsample(nn.Module):
    """
    Upsamples using pixel shuffle technique after increasing channel dimensions.
    """

    def __init__(self, in_channels, out_channels, upscale_factor=2):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels,
            out_channels * (upscale_factor ** 2),
            kernel_size=3,
            padding=1
        )
        self.pixel_shuffle = nn.PixelShuffle(upscale_factor)
        self.norm = nn.GroupNorm(8, out_channels)
        self.act = nn.GELU()

    def forward(self, x):
        x = self.conv(x)  # (B, C*r^2, H, W)
        x = self.pixel_shuffle(x)  # (B, C, H*r, W*r)
        x = self.norm(x)
        x = self.act(x)
        return x



class Decoder(nn.Module):
    """
    Transformer-style decoder that converts text embeddings into an image tensor.

    The decoder uses a learned latent tensor, cross-attention with the input text embeddings,
    residual blocks, and progressive upsampling (via transposed convolutions and pixel shuffle)
    to produce an output image.

    Args:
        text_embed_dim (int): Dimensionality of the input text embeddings.
        latent_size (tuple): Shape of the learned latent tensor as (C, H, W).
        num_heads (int): Number of attention heads used in cross-attention layers.
        decoder_depth (int): Number of decoding stages, each with resblock, upsampling, and attention.
        output_size (tuple): Final output image size as (channels, height, width).
        seq_len (int): Length of the input token sequence from the text encoder.

    Inputs:
        encoder_output (torch.Tensor): Text encoder output of shape (batch_size, seq_len, text_embed_dim).

    Returns:
        torch.Tensor: Decoded image of shape (batch_size, output_size[0], output_size[1], output_size[2]).
    """

    def __init__(self, text_embed_dim, latent_size=(512, 8, 8), num_heads=8, decoder_depth=4, output_size=(3, 215, 215),
                 seq_len=168):
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

        self.resblocks = nn.ModuleList()
        self.attentions = nn.ModuleList()
        self.upscales = nn.ModuleList()

        for d in range(decoder_depth):
            self.resblocks.append(ResBlock(in_channels=current_channels, out_channels=current_channels))
            self.upscales.append(ConvTranspose2d(in_channels=current_channels, out_channels=current_channels // 2)
                                 if d != decoder_depth - 1 else PixelShuffleUpsample(current_channels,
                                                                                     current_channels // 2, ))
            current_channels = current_channels // 2
            self.attentions.append(
                CrossAttention(num_heads=num_heads, embed_dim=current_channels, map_dim=text_embed_dim))

        self.last_conv = nn.Sequential(ResBlock(in_channels=current_channels, out_channels=current_channels // 2),
                                       nn.Conv2d(current_channels // 2, 3, kernel_size=1))

    def forward(self, encoder_output):
        batch_size = encoder_output.shape[0]

        latent = self.latent.repeat(batch_size, 1, 1, 1)
        x = F.gelu(self.text_attention(encoder_output, latent))
        for res, attn, up in zip(self.resblocks, self.attentions, self.upscales):
            x = res(x)
            x = F.gelu(x)
            x = up(x)
            x = attn(encoder_output, x)  # pass latent here
        x = F.tanh(self.last_conv(x))
        if x.shape[1:] != self.output_size:
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
                      decoder_depth=model_config['decoder_depth'], output_size=model_config['output_size'])

    out = decoder(torch.rand(8, 168, 256))

    summary(decoder, input_size=(4, 168, 256))
