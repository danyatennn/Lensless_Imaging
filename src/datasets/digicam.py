from pathlib import Path

import numpy as np
import torch
from huggingface_hub import hf_hub_download
from tqdm.auto import tqdm

from datasets import load_dataset
from lensless_helpers.preprocessor import (
    convert_image_to_float,
    force_rgb,
    get_cropped_lensed,
)
from lensless_helpers.psf import simulate_psf_from_mask
from src.datasets.base_dataset import BaseDataset

REPO_ID = "bezzam/DigiCam-Mirflickr-MultiMask-10K"


class DigiCamMirflickr(BaseDataset):

    def __init__(
        self, part="train", limit=None, shuffle_index=False, cache_dir=None, **kwargs
    ):
        self._hf_cache_dir = cache_dir
        self._data = load_dataset(REPO_ID, split=part, cache_dir=cache_dir)
        mask_labels = self._data["mask_label"]

        index = [
            {"row": i, "mask_label": int(mask_labels[i])}
            for i in range(len(self._data))
        ]

        self._psf_cache = self._build_psf_cache(
            sorted(set(int(label) for label in mask_labels)), cache_dir, part
        )

        super().__init__(index, limit=limit, shuffle_index=shuffle_index, **kwargs)

    def _build_psf_cache(self, unique_labels, cache_dir, part):
        psf_dir = None
        if cache_dir is not None:
            psf_dir = Path(cache_dir) / "psf_cache"
            psf_dir.mkdir(parents=True, exist_ok=True)

        cache = {}
        to_simulate = []
        for label in unique_labels:
            pt_path = psf_dir / f"psf_{label}.pt" if psf_dir else None
            if pt_path is not None and pt_path.exists():
                cache[label] = torch.load(pt_path, weights_only=True)
            else:
                to_simulate.append((label, pt_path))

        for label, pt_path in tqdm(
            to_simulate, desc=f"Simulating {part} PSFs", disable=not to_simulate
        ):
            psf = simulate_psf_from_mask(self._load_mask(label)).float().detach()
            cache[label] = psf
            if pt_path is not None:
                torch.save(psf, pt_path)

        return cache

    def _load_mask(self, label):
        path = hf_hub_download(
            repo_id=REPO_ID,
            filename=f"masks/mask_{label}.npy",
            repo_type="dataset",
            cache_dir=self._hf_cache_dir,
        )
        return np.load(path)

    def __getitem__(self, ind):
        meta = self._index[ind]
        row = self._data[meta["row"]]

        lensless = convert_image_to_float(force_rgb(np.array(row["lensless"])))
        lensless = torch.rot90(torch.from_numpy(lensless), dims=(-3, -2), k=2)

        lensed = convert_image_to_float(force_rgb(np.array(row["lensed"])))
        lensed = torch.from_numpy(get_cropped_lensed(lensed, lensless))

        psf = self._psf_cache[meta["mask_label"]]

        instance = {
            "measurement": lensless.permute(2, 0, 1).contiguous(),
            "gt": lensed.permute(2, 0, 1).contiguous(),
            "psf": psf[0].permute(2, 0, 1).contiguous(),
            "id": str(meta["row"]),
        }
        return self.preprocess_data(instance)

    @staticmethod
    def _assert_index_is_valid(index):
        for entry in index:
            assert "row" in entry, "Each dataset item should include field 'row'"
