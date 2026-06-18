from pathlib import Path

import numpy as np
import torch
from PIL import Image

from lensless_helpers.preprocessor import (
    convert_image_to_float,
    force_rgb,
    get_cropped_lensed,
)
from lensless_helpers.psf import simulate_psf_from_mask
from src.datasets.base_dataset import BaseDataset


class CustomDirDataset(BaseDataset):

    def __init__(self, data_dir, limit=None, shuffle_index=False, **kwargs):
        data_dir = Path(data_dir)
        lensless_dir = data_dir / "lensless"
        masks_dir = data_dir / "masks"
        lensed_dir = data_dir / "lensed"

        index = []
        for lensless_path in sorted(lensless_dir.glob("*.png")):
            sample_id = lensless_path.stem
            entry = {
                "id": sample_id,
                "lensless": str(lensless_path),
                "mask": str(masks_dir / f"{sample_id}.npy"),
            }
            lensed_path = lensed_dir / f"{sample_id}.png"
            if lensed_path.exists():
                entry["lensed"] = str(lensed_path)
            index.append(entry)

        self._psf_cache = {}
        super().__init__(index, limit=limit, shuffle_index=shuffle_index, **kwargs)

    def _get_psf(self, mask_path):
        if mask_path not in self._psf_cache:
            mask = np.load(mask_path)
            self._psf_cache[mask_path] = simulate_psf_from_mask(mask).float().detach()
        return self._psf_cache[mask_path]

    def __getitem__(self, ind):
        meta = self._index[ind]

        lensless = convert_image_to_float(
            force_rgb(np.array(Image.open(meta["lensless"]).convert("RGB")))
        )
        lensless = torch.rot90(torch.from_numpy(lensless), dims=(-3, -2), k=2)

        psf = self._get_psf(meta["mask"])

        instance = {
            "measurement": lensless.permute(2, 0, 1).contiguous(),
            "psf": psf[0].permute(2, 0, 1).contiguous(),
            "id": meta["id"],
        }

        if "lensed" in meta:
            lensed = convert_image_to_float(
                force_rgb(np.array(Image.open(meta["lensed"]).convert("RGB")))
            )
            lensed = torch.from_numpy(get_cropped_lensed(lensed, lensless))
            instance["gt"] = lensed.permute(2, 0, 1).contiguous()

        return self.preprocess_data(instance)

    @staticmethod
    def _assert_index_is_valid(index):
        for entry in index:
            assert "lensless" in entry, "Each item should include field 'lensless'"
            assert "mask" in entry, "Each item should include field 'mask'"
