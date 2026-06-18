import torch
from torch import nn


class ModularReconstruction(nn.Module):

    def __init__(
        self,
        camera_inversion,
        pre_processor=None,
        post_processor=None,
        normalize=True,
    ):
        super().__init__()
        self.pre_processor = pre_processor
        self.camera_inversion = camera_inversion
        self.post_processor = post_processor
        self.normalize = normalize

    @staticmethod
    def _normalize(x):
        max_val = torch.amax(x, dim=(-3, -2, -1), keepdim=True)
        return x / (max_val + 1e-6)

    def forward(self, measurement, psf, **batch):
        x = measurement
        if self.pre_processor is not None:
            x = self.pre_processor(x)

        x = self.camera_inversion(x, psf)

        if self.normalize:
            x = self._normalize(x)

        if self.post_processor is not None:
            x = self.post_processor(x)

        return {"reconstruction": x}

    def __str__(self):
        all_parameters = sum(p.numel() for p in self.parameters())
        trainable_parameters = sum(
            p.numel() for p in self.parameters() if p.requires_grad
        )
        result_info = super().__str__()
        result_info = result_info + f"\nAll parameters: {all_parameters}"
        result_info = result_info + f"\nTrainable parameters: {trainable_parameters}"
        return result_info
