import lpips
import torch
from torch import nn

from src.utils.lensless import crop_roi


class MSELPIPSLoss(nn.Module):

    def __init__(self, mse_weight=1.0, lpips_weight=1.0, net="vgg"):
        super().__init__()
        self.mse_weight = mse_weight
        self.lpips_weight = lpips_weight
        self.mse = nn.MSELoss()
        self.lpips = lpips.LPIPS(net=net)
        for param in self.lpips.parameters():
            param.requires_grad = False

    def forward(self, reconstruction, gt, **batch):
        rec = crop_roi(reconstruction)
        target = crop_roi(gt)

        mse = self.mse(rec, target)
        lpips_value = self.lpips(
            rec.clamp(0, 1) * 2 - 1, target.clamp(0, 1) * 2 - 1
        ).mean()

        loss = self.mse_weight * mse + self.lpips_weight * lpips_value
        return {"loss": loss, "mse_loss": mse, "lpips_loss": lpips_value}
