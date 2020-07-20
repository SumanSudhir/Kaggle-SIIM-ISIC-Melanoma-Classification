import torch
import torch.nn as nn
import torchvision.models as models


class ResNetModel(nn.Module):
    def __init__(self, out_dim=1):
        super().__init__()
        self.backbone = models.resnet18(pretrained=True)
        nc = list(self.backbone.children())[-1].in_features
        self.backbone = nn.Sequential(*list(self.backbone.children())[:-2])

        self.ap = nn.AdaptiveAvgPool2d((1,1))
        self.mp = nn.AdaptiveMaxPool2d((1,1))
        self.flatten = nn.Flatten()
        self.linear = nn.Linear(2*nc, out_dim)
        self.dropout_fc = nn.Dropout(0.10)

    def forward(self, x):
        x = self.backbone(x)
        x = torch.cat([self.mp(x), self.ap(x)], 1)
        x = self.flatten(x)
        x = self.linear(x)
        x = self.dropout_fc(x)

        return x
