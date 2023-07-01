# 基于VanillaNet的backbone网络，用于SiamFC网络

from .VanillaNet.models.vanillanet import vanillanet_5
import torch.nn as nn

class VanillaNet(nn.Module):
    def __init__(self):
        super(VanillaNet, self).__init__()
        self.conv1 = vanillanet_5()
        self.conv2 = nn.Sequential(
            nn.Conv2d(256, 256, 3, 1, groups=2),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True))
        self.conv3 = nn.Sequential(
            nn.Conv2d(256, 256, 3, 1, groups=2))

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        return x
