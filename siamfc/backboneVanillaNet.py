# 基于VanillaNet的backbone网络，用于SiamFC网络

from .VanillaNet.models.vanillanet import vanillanet_5
import torch.nn as nn

# vanillanet_5 = vanillanet_5()
# #修改网络最后一层的输出通道数，使得输出通道数为256，与SiamFC网络的特征通道数一致，方便后续的特征提取，同时保证网络的输出尺寸不变
# vanillanet_5.conv5 = nn.Conv2d(128, 256, 3, 1, groups=2)

class VanillaNet(nn.Module):
    def __init__(self):
        super(VanillaNet, self).__init__()
        self.features = vanillanet_5()
        #修改网络最后一层的输出通道数，使得输出通道数为256，与SiamFC网络的特征通道数一致，方便后续的特征提取，同时保证网络的输出尺寸不变
        self.features.conv5 = nn.Conv2d(128, 256, 3, 1, groups=2)

    def forward(self, x):
        return self.features(x)