import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import vgg16
from torchvision.models.feature_extraction import create_feature_extractor


class PerceptualLoss(nn.Module):
    def __init__(self, layer_weights=None):
        super().__init__()

        vgg = vgg16(weights='DEFAULT').features.eval()  # Use pretrained VGG16
        for param in vgg.parameters():
            param.requires_grad = False  # Freeze weights

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
