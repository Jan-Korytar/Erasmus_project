
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import vgg16
from torchvision.models.feature_extraction import create_feature_extractor
from transformers import CLIPProcessor, CLIPModel


class PerceptualLoss(nn.Module):
    def __init__(self, layer_weights=None):
        super().__init__()

        vgg = vgg16(weights='DEFAULT').features.eval()
        for param in vgg.parameters():
            param.requires_grad = False

        # Select layers to extract features from
        self.extractor = create_feature_extractor(vgg, return_nodes={
            '3': 'relu1_2',  # conv1_2
            '8': 'relu2_2',  # conv2_2
            '15': 'relu3_3',  # conv3_3
            '22': 'relu4_3',  # conv4_3
        })

        self.layer_weights = layer_weights or {
            'relu1_2': 1.0,
            'relu2_2': 0.75,
            'relu3_3': 0.5,
            'relu4_3': 0.25,
        }

    def forward(self, pred, target):
        """
        image: (B, 3, H, W), range [-1, 1]
        text: list of strings (length B)
        """
        # Normalize input to match ImageNet stats
        mean = torch.tensor([0.485, 0.456, 0.406], device=pred.device).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=pred.device).view(1, 3, 1, 1)

        pred_norm = (pred + 1) / 2  # scale from [-1, 1] → [0, 1]
        target_norm = (target + 1) / 2

        pred_norm = (pred_norm - mean) / std
        target_norm = (target_norm - mean) / std

        pred_feats = self.extractor(pred_norm)
        target_feats = self.extractor(target_norm)

        loss = 0.0
        for key in self.layer_weights:
            loss += self.layer_weights[key] * F.mse_loss(pred_feats[key], target_feats[key])
        return loss


class CLIPLoss(torch.nn.Module):
    def __init__(self, device='cuda'):
        super().__init__()
        self.device = device
        self.model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32", use_fast=False)

    def forward(self, image, text):
        """
        image: (B, 3, H, W), range [-1, 1]
        text: list of strings (length B)
        """
        # Remove special tokens
        for i in range(len(text)):
            text[i] = ' '.join([word for word in text[i].split(' ') if word not in ['[MASK]', '[NAME]']])

        image = (image + 1) / 2

        inputs = self.processor(text=text, images=image, return_tensors="pt", padding=True, truncation=True,
                                do_rescale=False).to(self.device)

        outputs = self.model(**inputs)
        image_features = outputs.image_embeds
        text_features = outputs.text_embeds

        image_features = F.normalize(image_features, dim=-1)
        text_features = F.normalize(text_features, dim=-1)

        loss = 1 - (image_features * text_features).sum(dim=-1).mean()
        return loss


class LatentDecorrelationLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        """
        Args:
            latent: Tensor of shape (B=1, C, H, W),
        Returns:
            Scalar loss encouraging feature decorrelation across channels.
        """
        b, c, h, w = latent.shape
        x = latent.view(c, -1)  # shape: (C, H*W)

        x = x - x.mean(dim=1, keepdim=True)
        cov = x @ x.T / x.shape[1]  # shape: (C, C)

        off_diag = cov - torch.diag(torch.diag(cov))  # zero out diagonal
        loss = off_diag.pow(2).mean()
        return loss


class ColorMomentLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, output_rgb, target_rgb):
        """
        Args:
            output_rgb: Tensor (B, 3, H, W),
            target_rgb: Tensor (B, 3, H, W),
        Returns:
            Scalar loss: L1 loss between color means and stds per channel.
        """
        # Compute mean per channel
        output_rgb = (output_rgb + 1) / 2
        target_rgb = (target_rgb + 1) / 2
        output_mean = output_rgb.mean(dim=[2, 3])
        target_mean = target_rgb.mean(dim=[2, 3])
        mean_loss = F.l1_loss(output_mean, target_mean)

        # Compute std per channel
        output_std = output_rgb.std(dim=[2, 3])
        target_std = target_rgb.std(dim=[2, 3])
        std_loss = F.l1_loss(output_std, target_std)

        return mean_loss + std_loss
