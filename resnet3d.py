import torch
import torch.nn as nn
import torchvision.models.video as video_models

import torch
import torch.nn as nn
# import models_vit
import torchvision.models.video as video_models

class ResNet3DEncoder(nn.Module):
    """
    Frozen 3D ResNet encoder that outputs a 512-dimensional feature vector per volume.
    - Backbone (r3d_18/mc3_18/r2plus1d_18) is frozen (no grad).
    - Only the new projection head is trainable.

    Input: x of shape (B, C, D, H, W)
    Output: features of shape (B, 512)
    """
    def __init__(
        self,
        backbone: str = 'r3d_18',
        pretrained: bool = True,
        dropout: float = 0.0,
    ):
        super().__init__()
        # 1) Load backbone
        if backbone == 'r3d_18':
            model = video_models.r3d_18(pretrained=pretrained)
        elif backbone == 'mc3_18':
            model = video_models.mc3_18(pretrained=pretrained)
        elif backbone == 'r2plus1d_18':
            model = video_models.r2plus1d_18(pretrained=pretrained)
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")

        # 2) Freeze all backbone parameters
        for param in model.parameters():
            param.requires_grad = False

        # 3) Replace final fc with a new projection head
        in_features = model.fc.in_features  # usually 512
        proj_head = []
        if dropout > 0:
            proj_head.append(nn.Dropout(dropout))
        proj_head += [
            nn.Linear(in_features, 512),  # project to 512-d
            nn.ReLU(inplace=True)
        ]
        self.encoder = model
        self.encoder.fc = nn.Sequential(*proj_head)

        # Note: this new fc layer's parameters default to requires_grad=True.

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape (B, C, D, H, W)
        Returns:
            Tensor of shape (B, 512)
        """
        # torchvision video models expect input shape (B, C, T, H, W)
        return self.encoder(x)