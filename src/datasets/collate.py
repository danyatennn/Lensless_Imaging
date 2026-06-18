import torch


def collate_fn(dataset_items: list[dict]):
    result_batch = {
        "measurement": torch.stack([item["measurement"] for item in dataset_items]),
        "psf": torch.stack([item["psf"] for item in dataset_items]),
        "id": [item["id"] for item in dataset_items],
    }

    if "gt" in dataset_items[0]:
        result_batch["gt"] = torch.stack([item["gt"] for item in dataset_items])

    return result_batch
