import time
import warnings

import hydra
import torch
from hydra.utils import instantiate

from src.utils.init_utils import get_device, setup_writer

warnings.filterwarnings("ignore", category=UserWarning)


@hydra.main(version_base=None, config_path="src/configs", config_name="inference")
def main(config):
    device = get_device(config.inferencer.device)
    n_warmup = config.get("n_warmup", 3)
    n_runs = config.get("n_runs", 20)
    shape = (1, 3, 380, 507)

    model = instantiate(config.model).to(device).eval()

    measurement = torch.rand(*shape, device=device)
    psf = torch.rand(*shape, device=device)
    psf = psf / psf.flatten(1).norm(dim=1).view(-1, 1, 1, 1)

    with torch.no_grad():
        for _ in range(n_warmup):
            model(measurement=measurement, psf=psf)

        if device == "cuda":
            torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(n_runs):
            model(measurement=measurement, psf=psf)
        if device == "cuda":
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - start

    avg_ms = elapsed / n_runs * 1000
    print(f"Device: {device}")
    print(f"Average reconstruction time: {avg_ms:.1f} ms")

    writer = setup_writer(config)
    if writer is not None:
        writer.set_step(0, mode="speed")
        writer.add_scalar("reconstruction_time_ms", avg_ms)


if __name__ == "__main__":
    main()
