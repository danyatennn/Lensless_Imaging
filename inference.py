import warnings
from pathlib import Path

import hydra
from hydra.utils import instantiate

from src.datasets.data_utils import get_dataloaders
from src.trainer import Inferencer
from src.utils.init_utils import get_device, set_random_seed, setup_writer
from src.utils.io_utils import ROOT_PATH

warnings.filterwarnings("ignore", category=UserWarning)


@hydra.main(version_base=None, config_path="src/configs", config_name="inference")
def main(config):
    """
    Main script for inference. Instantiates the model, metrics, and
    dataloaders. Runs Inferencer to calculate metrics and save predictions.

    Args:
        config (DictConfig): hydra experiment config.
    """
    set_random_seed(config.inferencer.seed)

    device = get_device(config.inferencer.device)

    dataloaders, batch_transforms = get_dataloaders(config, device)

    model = instantiate(config.model).to(device)
    print(model)

    metrics = instantiate(config.metrics)

    save_path = Path(config.inferencer.save_path)
    if not save_path.is_absolute():
        save_path = ROOT_PATH / "data" / "saved" / save_path
    save_path.mkdir(exist_ok=True, parents=True)

    skip_model_load = config.inferencer.get("from_pretrained") is None

    writer = setup_writer(config)

    inferencer = Inferencer(
        model=model,
        config=config,
        device=device,
        dataloaders=dataloaders,
        batch_transforms=batch_transforms,
        save_path=save_path,
        metrics=metrics,
        skip_model_load=skip_model_load,
        writer=writer,
    )

    logs = inferencer.run_inference()

    for part in logs.keys():
        for key, value in logs[part].items():
            full_key = part + "_" + key
            print(f"    {full_key:15s}: {value}")


if __name__ == "__main__":
    main()
