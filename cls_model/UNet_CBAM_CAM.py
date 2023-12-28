import torch
import torch.nn as nn
from UNet_CBAM import UNet_CBAM


class UNet_CBAM_CAM(nn.Module):
    def __init__(self, in_channel, out_channel, classes):
        super(UNet_CBAM_CAM, self).__init__()

        self.unet = UNet_CBAM(in_channels=in_channel, out_channels=out_channel)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(32, classes, bias=True)

    def forward(self, x):
        unet_out = self.unet(x)
        pooled = self.avg_pool(unet_out)
        pooled = pooled.view(pooled.size(0), -1)
        fc_out = self.fc(pooled)

        return fc_out
