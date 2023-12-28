import monai
import torch
import torch.nn as nn
from ssim_loss import SSIM


class Myloss(nn.Module):
    def __init__(self, issigmoid=True):
        super().__init__()
        self.issigmoid = issigmoid
        self.dice_loss = monai.losses.DiceLoss(sigmoid=issigmoid, squared_pred=True, reduction='mean')
        # self.ssim_loss = monai.losses.SSIMLoss(win_size=11, spatial_dims=2)
        self.ssim_loss = SSIM(window_size=11, size_average=True)
        self.bce_loss = nn.BCELoss(reduction='mean')

    def forward(self, pred, gt):
        dice = self.dice_loss(pred, gt)
        if self.issigmoid:
            pred = torch.sigmoid(pred)
        # ssim = self.ssim_loss(pred, gt)
        ssim = 1 - self.ssim_loss(pred, gt)
        bce = self.bce_loss(pred, gt)

        # loss = dice + bce
        loss = dice + ssim + bce

        return loss