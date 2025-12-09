import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18


class ResNet18CIFAR(nn.Module):
    """
    ResNet-18 adapted for CIFAR-10.

    RL will choose how many of these "layers" to train:
      [0] stem (conv1+bn1+relu)
      [1] layer1
      [2] layer2
      [3] layer3
      [4] layer4
      [5] classifier (avgpool + fc)
    So num_layers_for_rl = 6  (depth in [1..6])
    """

    def __init__(self, num_classes=10):
        super().__init__()

        # base ResNet-18
        backbone = resnet18(weights=None)  # torchvision>=0.13, otherwise use pretrained=False

        # Modify for CIFAR-10: smaller images → smaller conv, no maxpool
        backbone.conv1 = nn.Conv2d(
            3, 64, kernel_size=3, stride=1, padding=1, bias=False
        )
        backbone.maxpool = nn.Identity()
        backbone.fc = nn.Linear(backbone.fc.in_features, num_classes)

        self.stem = nn.Sequential(
            backbone.conv1,
            backbone.bn1,
            backbone.relu,
        )
        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        self.layer4 = backbone.layer4
        self.avgpool = backbone.avgpool
        self.fc = backbone.fc

        # Expose blocks + classifier in the same way as your previous CNN
        self.blocks = nn.ModuleList([
            self.stem,
            self.layer1,
            self.layer2,
            self.layer3,
            self.layer4,
        ])
        self.classifier = nn.Sequential(
            self.avgpool,
            nn.Flatten(),
            self.fc,
        )

    def forward(self, x):
        for b in self.blocks:
            x = b(x)
        x = self.classifier(x)
        return x

    @property
    def num_layers_for_rl(self):
        # 5 blocks + 1 classifier = 6 RL "layers"
        return len(self.blocks) + 1
