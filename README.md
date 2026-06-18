# Lensless Computational Imaging

Reconstruction of images from lensless camera measurements. This
repository reproduces and compares reconstruction
algorithms on the
[DigiCam-Mirflickr-MultiMask-10K](https://huggingface.co/datasets/bezzam/DigiCam-Mirflickr-MultiMask-10K)
dataset base on these articles:

- [Towards Robust and Generalizable Lensless Imaging with Modular Learned Reconstruction](https://arxiv.org/abs/2502.01102)
- [Learned reconstructions for practical mask-based lensless imaging](https://arxiv.org/abs/1908.11502)

The project is built on the
[PyTorch Project Template](https://github.com/Blinorot/pytorch_project_template).

## Contents

- [Methods](#methods)
- [Installation](#installation)
- [Dataset](#dataset)
- [Training](#training)
- [Inference and metrics](#inference-and-metrics)
- [Reconstruction speed](#reconstruction-speed)
- [Checkpoints](#checkpoints)
- [Demo](#demo)


## Methods

| Config | Pre | Inversion | Post | Trainable params |
|---|---|---|---|---|
| [`admm100`](src/configs/model/admm100.yaml) | – | ADMM, 100 iters, fixed `mu=1e-4, tau=2e-4` | – | 0 |
| [`unrolled_admm20`](src/configs/model/unrolled_admm20.yaml) | – | Unrolled ADMM, 20 iters, learnable | – | 80 |
| [`modular_pre_post`](src/configs/model/modular_pre_post.yaml) | DRUNet 4.1M | Unrolled ADMM, 5 iters | DRUNet 4.1M | ~8.1M |
| [`modular_pre`](src/configs/model/modular_pre.yaml) | DRUNet 8.2M | Unrolled ADMM, 5 iters | – | ~8.2M |
| [`modular_post`](src/configs/model/modular_post.yaml) | – | Unrolled ADMM, 5 iters | DRUNet 8.2M | ~8.2M |

- Training loss is `MSE + LPIPS (VGG)` computed on the region of interest (ROI)
- Metrics are PSNR, SSIM, LPIPS (VGG), and MSE also computed on the ROI

### Bonus methods

| Config | Description | Trainable params |
|---|---|---|
| [`fista`](src/configs/model/fista.yaml) | FISTA, 100 iters, fixed (non-ADMM solver) | 0 |
| [`fista_unrolled`](src/configs/model/fista_unrolled.yaml) | Unrolled FISTA, 20 iters, learnable | 40 |
| [`admm100_realesrgan`](src/configs/model/admm100_realesrgan.yaml) | ADMM-100 + frozen pretrained Real-ESRGAN | 0 |
| [`admm100_realesrgan_ft`](src/configs/model/admm100_realesrgan_ft.yaml) | ADMM-100 + fine-tuned Real-ESRGAN | 16.7M |

## Installation

```bash
# (optional) create an environment
python3 -m venv project_env
source project_env/bin/activate

# install dependencies
pip install -r requirements.txt

# (optional) install pre-commit hooks
pre-commit install
```

The device is chosen automatically: CUDA if available, then MPS, then CPU

## Dataset

The training and evaluation data are the `train` and `test` splits of
[DigiCam-Mirflickr-MultiMask-10K](https://huggingface.co/datasets/bezzam/DigiCam-Mirflickr-MultiMask-10K).
The dataset is downloaded automatically on first use

PSFs are stored as mask patterns and simulated from the masks (with
waveprop)

## Training

```bash
python3 train.py -cn=lensless model=MODEL writer.run_name=RUN_NAME
```

`MODEL` is one of `unrolled_admm20`, `modular_pre_post`, `modular_pre`,
`modular_post` (the `admm100` baseline is parameter-free and is only used for
inference). Examples:

```bash
python3 train.py -cn=lensless model=unrolled_admm20  writer.run_name=unrolled_admm20
python3 train.py -cn=lensless model=modular_pre_post writer.run_name=modular_pre_post
python3 train.py -cn=lensless model=modular_pre      writer.run_name=modular_pre
python3 train.py -cn=lensless model=modular_post     writer.run_name=modular_post
```

Bonus learnable models (see [Bonus methods](#bonus-methods)):

```bash
# unrolled FISTA (non-ADMM solver)
python3 train.py -cn=lensless model=fista_unrolled writer.run_name=fista_unrolled

# fine-tune Real-ESRGAN on top of fixed ADMM-100
python3 train.py -cn=lensless model=admm100_realesrgan_ft \
    optimizer.lr=1e-5 trainer.n_epochs=10 dataloader.batch_size=2 \
    writer.run_name=realesrgan_ft
```

- `trainer.n_epochs=N` — number of epochs (default 25)
- `dataloader.batch_size=B` — (default 4)
- `datasets.test.limit=K` — number of test images used for per-epoch

Metrics, losses, learning rate, and reconstruction examples are logged every
`trainer.log_step` steps and after every epoch. Checkpoints are saved every
`trainer.save_period` epochs and the best one (by `test_PSNR`) is saved as
`saved/RUN_NAME/model_best.pth`

Logging requires a Comet ML API key (set the `COMET_API_KEY` environment)

## Inference and metrics

Run a trained model on a dataset and save the reconstructions:

```bash
python3 inference.py \
    model=modular_pre_post \
    inferencer.from_pretrained=saved/modular_pre_post/model_best.pth \
    inferencer.save_path=data/saved/modular_pre_post
```

For the classic ADMM-100 baseline there is no checkpoint:

```bash
python3 inference.py \
    model=admm100 \
    inferencer.from_pretrained=null \
    inferencer.save_path=data/saved/admm100
```

Bonus methods. FISTA and the frozen ADMM-100 + Real-ESRGAN have no checkpoint
the fine-tuned Real-ESRGAN:

```bash
python3 inference.py model=fista inferencer.from_pretrained=null \
    inferencer.save_path=data/saved/fista
python3 inference.py model=admm100_realesrgan inferencer.from_pretrained=null \
    inferencer.save_path=data/saved/admm100_realesrgan
python3 inference.py model=admm100_realesrgan_ft \
    inferencer.from_pretrained=saved/realesrgan_ft/model_best.pth \
    inferencer.save_path=data/saved/realesrgan_ft
```

Run on a custom directory (see [Demo](#demo)):

```bash
python3 inference.py \
    model=modular_pre_post \
    inferencer.from_pretrained=saved/modular_pre_post/model_best.pth \
    datasets=custom_dir datasets.test.data_dir=/path/to/data \
    inferencer.save_path=/path/to/reconstructions
```

Reconstructions are saved as `ID.png` (the ROI) matching the input image id.

Inference logging to Comet ML is automatic but optional: if the `COMET_API_KEY`
is set, the final metrics and reconstruction examples
are logged to Comet ML

```bash
export COMET_API_KEY=your_key
python3 inference.py model=modular_pre_post \
    inferencer.from_pretrained=saved/modular_pre_post/model_best.pth \
    inferencer.save_path=data/saved/modular_pre_post writer.run_name=modular_pre_post
```

Compute metrics between ground-truth and reconstructed images:

```bash
python3 calculate_metrics.py \
    --ground_truth /path/to/data \
    --reconstructions /path/to/reconstructions
```

The `--ground_truth` argument is a directory of images or a data
directory containing a `lensed/` subdir. It prints metrics

## Reconstruction speed

```bash
python3 measure_speed.py model=admm100
python3 measure_speed.py model=modular_pre_post
```

It prints and logges the average reconstruction time

## Checkpoints

Final checkpoints are hosted on Google Drive. They are already set in
[`download_checkpoints.py`](download_checkpoints.py):

```bash
python3 download_checkpoints.py --model all
```

Checkpoints are saved to `saved/<model>/model_best.pth`.

## Demo

[`demo.ipynb`](demo.ipynb) clones the repository, installs dependencies, downloads the
checkpoints, runs inference on a custom `.zip` dataset (Google Drive
link), visualizes samples, and computes the metrics

