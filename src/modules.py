import torch
import torch.nn as nn
import torchvision.models as models

from efficientnet_pytorch import EfficientNet
from utils import AdaptiveConcatPool2d

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
        self.dropout_fc = nn.Dropout(0.1)

    def forward(self, x):
        x = self.backbone(x)
        x = torch.cat([self.mp(x), self.ap(x)], 1)
        x = self.flatten(x)
        x = self.linear(x)
        x = self.dropout_fc(x)

        return x


class EfficientModel(nn.Module):
    def __init__(self, n_meta_features, num_class=1):
        super().__init__()
        self.enet = EfficientNet.from_pretrained('efficientnet-b0')

        self.dropout_fc = nn.Dropout(0.2)

        self.head = nn.Sequential(AdaptiveConcatPool2d(),
                                  nn.Flatten(),
                                  nn.Linear(2*self.enet._fc.in_features, 512),
                                  nn.BatchNorm1d(512),
                                  nn.ReLU(),
                                  nn.Dropout(p=0.2))

        self.meta = nn.Sequential(nn.Linear(n_meta_features, 512),
                                  nn.BatchNorm1d(512),
                                  nn.ReLU(),
                                  nn.Dropout(p=0.2),
                                  nn.Linear(512, 256),
                                  nn.BatchNorm1d(256),
                                  nn.ReLU(),
                                  nn.Dropout(p=0.2))

        self.classifier = nn.Sequential(nn.Linear(512 + 256, 256),
                                  nn.BatchNorm1d(256),
                                  nn.ReLU(),
                                  nn.Dropout(p=0.2),
                                  nn.Linear(256, num_class))


    def forward(self, x, y):
        x = self.enet.extract_features(x)

        x = self.head(x)
        y = self.meta(y)

        features = torch.cat((x, y), dim=1)

        out = self.classifier(features)
        out = self.dropout_fc(out)

        return out
