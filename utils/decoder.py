import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from torch.ao.nn.quantized import ConvTranspose2d
from torchinfo import summary

from utils.helpers import get_project_root


#latent space 256, 8, 8

class CrossAttention(nn.Module):
    """
    Applies cross-attention between decoder latent and encoder output.
    """

    def __init__(self, num_heads=8, embed_query_dim=None, vdim_kdim=256, H_W: int = None):
        super().__init__()
        if vdim_kdim is None:
            vdim_kdim = embed_query_dim
        self.pos_embed = nn.Parameter(torch.randn(1, embed_query_dim, H_W, H_W))
        self.attention = nn.MultiheadAttention(num_heads=num_heads, embed_dim=embed_query_dim, vdim=vdim_kdim,
                                               kdim=vdim_kdim, batch_first=True, dropout=0.1)
        self.norm = nn.LayerNorm(embed_query_dim)

    def forward(self, encoder_output, latent):
        B, C, H, W = latent.shape
        latent_pos = latent + self.pos_embed
        latent_pos = latent_pos.view(B, C, H * W).permute(0, 2, 1)  # -> B, HW, C
        attn_output, _ = self.attention(query=latent_pos, key=encoder_output, value=encoder_output)
        output = self.norm(latent_pos + attn_output)
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

    def __init__(self, in_channels, out_channels, num_groups=8):
        super().__init__()
        self.block = nn.Sequential(
            nn.GroupNorm(num_groups, in_channels),
            nn.GELU(),
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.GroupNorm(num_groups, out_channels),
            nn.GELU(),
            nn.Conv2d(out_channels, out_channels, 3, padding=1)
        )
        self.skip = nn.Conv2d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()

    def forward(self, x):
        return self.block(x) + self.skip(x)


class ConvTranspose2d(nn.Module):
    """
    Applies transposed convolution  for upsampling.
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

    def __init__(self, in_channels, out_channels, num_groups=8, upscale_factor=2):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels,
            out_channels * (upscale_factor ** 2),
            kernel_size=3,
            padding=1
        )
        self.pixel_shuffle = nn.PixelShuffle(upscale_factor)
        self.norm = nn.GroupNorm(num_groups, out_channels)
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
        decay_channels (float): How much to lower the channel size layer by layer. default: 0.75

    Inputs:
        encoder_output (torch.Tensor): Text encoder output of shape (batch_size, seq_len, text_embed_dim).

    Returns:
        torch.Tensor: Decoded image of shape (batch_size, *output_size).
    """

    def __init__(self, text_embed_dim, latent_size=(512, 8, 8), num_heads=8, decoder_depth=4, output_size=(3, 215, 215),
                 decay_channels=0.75):
        super().__init__()
        self.output_size = output_size
        self.latent_size = latent_size
        self.text_embed_dim = text_embed_dim
        self.latent = nn.Parameter(torch.randn(1, *latent_size))
        self.depth = decoder_depth

        current_channels = latent_size[0]

        # self.z_dim = 256 was used to add some stochasticity.
        # self.z_to_latent = nn.Linear(self.z_dim, latent_size[0] * latent_size[1] * latent_size[2])

        # CrossAttention between latent and text
        self.text_attention = CrossAttention(num_heads=num_heads, embed_query_dim=current_channels,
                                             vdim_kdim=text_embed_dim, H_W=latent_size[-1])
        self.dropout = nn.Dropout(p=0.1)

        self.resblocks = nn.ModuleList()
        self.attentions = nn.ModuleList()
        self.upscales = nn.ModuleList()
        spatial_size = latent_size[-1]

        def closest_divisor(value, target=8):
            divisors = [d for d in range(1, (value // 2) + 1) if value % d == 0]
            closest = min(divisors, key=lambda x: abs(x - target))
            return int(closest)

        # Build the model layer by layer
        for d in range(decoder_depth):
            self.resblocks.append(
                ResBlock(in_channels=current_channels, out_channels=current_channels,
                         num_groups=current_channels // closest_divisor(current_channels, 8)))

            self.upscales.append(
                ConvTranspose2d(in_channels=current_channels, out_channels=int(current_channels * decay_channels))
                if d != decoder_depth - 1 else
                PixelShuffleUpsample(current_channels, int(current_channels * decay_channels),
                                     num_groups=int(current_channels * decay_channels) //
                                                closest_divisor(int(current_channels * decay_channels), 8)))
            current_channels = int(current_channels * decay_channels)
            spatial_size *= 2
            self.attentions.append(
                CrossAttention(num_heads=closest_divisor(current_channels, 8), embed_query_dim=current_channels,
                               vdim_kdim=text_embed_dim, H_W=spatial_size))
        # Channels to -> 3
        self.last_conv = nn.Sequential(
            ResBlock(in_channels=current_channels, out_channels=current_channels,
                     num_groups=current_channels // closest_divisor(current_channels, 8)),
            nn.Conv2d(current_channels, 3, kernel_size=1))


    def forward(self, encoder_output):
        batch_size = encoder_output.shape[0]
        latent = self.latent.repeat(batch_size, 1, 1, 1)
        # z = self.z_to_latent(torch.randn(batch_size, self.z_dim, device=latent.device)).view(*latent.shape)
        latent = latent  #+ z

        # adding stochasticity in training
        if self.training:
            latent = latent + torch.randn_like(latent) * 0.01
        x = F.gelu(self.text_attention(encoder_output, latent))  # x are the hiddens

        # core loop of the model
        for res, attn, up in zip(self.resblocks, self.attentions, self.upscales):
            x = res(x)
            x = self.dropout(F.gelu(x))
            x = up(x)
            x = attn(encoder_output, x)
        x = F.tanh(self.last_conv(x))

        # potentially make the output the right shape.
        if x.shape[1:] != self.output_size:
            x = F.interpolate(x, size=self.output_size[1:], mode="bilinear", align_corners=True)
        return x



if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    project_root = get_project_root()
    with open(project_root / 'config.yml') as f:
        config = yaml.safe_load(f)
        training_config = config['training']
        model_config = config['model']
    decoder = Decoder(text_embed_dim=256, latent_size=model_config['latent_size'],
                      decoder_depth=model_config['decoder_depth'], output_size=model_config['output_size'])

    out = decoder(torch.rand(8, 168, 256))

    summary(decoder, input_size=(4, 168, 256))
