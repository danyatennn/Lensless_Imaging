import torch
from torch import nn


def conv3x3(in_channels, out_channels):
    return nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False)


class ResBlock(nn.Module):

    def __init__(self, channels):
        super().__init__()
        self.block = nn.Sequential(
            conv3x3(channels, channels),
            nn.ReLU(inplace=True),
            conv3x3(channels, channels),
        )

    def forward(self, x):
        return x + self.block(x)


class DRUNet(nn.Module):

    def __init__(
        self,
        in_channels=3,
        out_channels=3,
        channels=(32, 64, 128, 256),
        n_res_blocks=4,
        residual=True,
    ):
        super().__init__()
        self.residual = residual
        c0, c1, c2, c3 = channels

        self.head = conv3x3(in_channels, c0)

        self.enc0 = nn.Sequential(*[ResBlock(c0) for _ in range(n_res_blocks)])
        self.down0 = nn.Conv2d(c0, c1, kernel_size=2, stride=2, bias=False)
        self.enc1 = nn.Sequential(*[ResBlock(c1) for _ in range(n_res_blocks)])
        self.down1 = nn.Conv2d(c1, c2, kernel_size=2, stride=2, bias=False)
        self.enc2 = nn.Sequential(*[ResBlock(c2) for _ in range(n_res_blocks)])
        self.down2 = nn.Conv2d(c2, c3, kernel_size=2, stride=2, bias=False)

        self.body = nn.Sequential(*[ResBlock(c3) for _ in range(n_res_blocks)])

        self.up2 = nn.ConvTranspose2d(c3, c2, kernel_size=2, stride=2, bias=False)
        self.dec2 = nn.Sequential(*[ResBlock(c2) for _ in range(n_res_blocks)])
        self.up1 = nn.ConvTranspose2d(c2, c1, kernel_size=2, stride=2, bias=False)
        self.dec1 = nn.Sequential(*[ResBlock(c1) for _ in range(n_res_blocks)])
        self.up0 = nn.ConvTranspose2d(c1, c0, kernel_size=2, stride=2, bias=False)
        self.dec0 = nn.Sequential(*[ResBlock(c0) for _ in range(n_res_blocks)])

        self.tail = conv3x3(c0, out_channels)

    @staticmethod
    def _pad_to_multiple(x, multiple=8):
        h, w = x.shape[-2:]
        pad_h = (multiple - h % multiple) % multiple
        pad_w = (multiple - w % multiple) % multiple
        return nn.functional.pad(x, (0, pad_w, 0, pad_h), mode="replicate")

    def forward(self, x):
        h, w = x.shape[-2:]
        padded = self._pad_to_multiple(x, multiple=8)

        e0 = self.enc0(self.head(padded))
        e1 = self.enc1(self.down0(e0))
        e2 = self.enc2(self.down1(e1))
        b = self.body(self.down2(e2))
        d2 = self.dec2(self.up2(b) + e2)
        d1 = self.dec1(self.up1(d2) + e1)
        d0 = self.dec0(self.up0(d1) + e0)
        out = self.tail(d0)[..., :h, :w]

        if self.residual:
            out = out + x
        return out
