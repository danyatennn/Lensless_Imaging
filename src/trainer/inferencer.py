import numpy as np
import torch
from PIL import Image
from tqdm.auto import tqdm

from src.metrics.tracker import MetricTracker
from src.trainer.base_trainer import BaseTrainer
from src.utils.lensless import crop_roi


class Inferencer(BaseTrainer):

    def __init__(
        self,
        model,
        config,
        device,
        dataloaders,
        save_path,
        metrics=None,
        batch_transforms=None,
        skip_model_load=False,
        writer=None,
    ):
        assert (
            skip_model_load or config.inferencer.get("from_pretrained") is not None
        ), "Provide checkpoint or set skip_model_load=True"

        self.config = config
        self.cfg_trainer = self.config.inferencer

        self.device = device

        self.model = model
        self.batch_transforms = batch_transforms
        self.writer = writer

        self.evaluation_dataloaders = {k: v for k, v in dataloaders.items()}

        self.save_path = save_path

        self.metrics = metrics
        if self.metrics is not None:
            self.evaluation_metrics = MetricTracker(
                *[m.name for m in self.metrics["inference"]],
                writer=None,
            )
        else:
            self.evaluation_metrics = None

        if not skip_model_load:
            self._from_pretrained(config.inferencer.get("from_pretrained"))

    def run_inference(self):
        part_logs = {}
        for part, dataloader in self.evaluation_dataloaders.items():
            logs = self._inference_part(part, dataloader)
            part_logs[part] = logs
        return part_logs

    def process_batch(self, batch_idx, batch, metrics, part):
        batch = self.move_batch_to_device(batch)
        batch = self.transform_batch(batch)

        outputs = self.model(**batch)
        batch.update(outputs)

        if metrics is not None and "gt" in batch:
            for met in self.metrics["inference"]:
                metrics.update(met.name, met(**batch))

        reconstruction = crop_roi(batch["reconstruction"]).clamp(0, 1)
        for i in range(reconstruction.shape[0]):
            image = reconstruction[i].permute(1, 2, 0).cpu().numpy()
            image = (image * 255).astype(np.uint8)
            Image.fromarray(image).save(self.save_path / f"{batch['id'][i]}.png")

        if self.writer is not None and batch_idx == 0:
            self._log_predictions(batch)

        return batch

    def _inference_part(self, part, dataloader):
        self.is_train = False
        self.model.eval()

        if self.evaluation_metrics is not None:
            self.evaluation_metrics.reset()

        self.save_path.mkdir(exist_ok=True, parents=True)

        if self.writer is not None:
            self.writer.set_step(0, mode=part)

        with torch.no_grad():
            for batch_idx, batch in tqdm(
                enumerate(dataloader), desc=part, total=len(dataloader)
            ):
                batch = self.process_batch(
                    batch_idx=batch_idx,
                    batch=batch,
                    part=part,
                    metrics=self.evaluation_metrics,
                )

        if self.evaluation_metrics is None:
            return {}

        result = self.evaluation_metrics.result()
        if self.writer is not None:
            for name, value in result.items():
                self.writer.add_scalar(name, value)
        return result

    def _log_predictions(self, batch, n_examples=4):
        n_examples = min(n_examples, batch["measurement"].shape[0])

        reconstruction = crop_roi(batch["reconstruction"]).clamp(0, 1)
        measurement = batch["measurement"]
        measurement = measurement / (
            measurement.amax(dim=(-3, -2, -1), keepdim=True) + 1e-6
        )
        gt = crop_roi(batch["gt"]).clamp(0, 1) if "gt" in batch else None

        for i in range(n_examples):
            sample_id = batch["id"][i]
            self.writer.add_image(
                f"measurement/{sample_id}", self._to_image(measurement[i])
            )
            self.writer.add_image(
                f"reconstruction/{sample_id}", self._to_image(reconstruction[i])
            )
            if gt is not None:
                self.writer.add_image(f"gt/{sample_id}", self._to_image(gt[i]))

    @staticmethod
    def _to_image(tensor):
        array = tensor.detach().clamp(0, 1).permute(1, 2, 0).cpu().numpy()
        return (array * 255).astype("uint8")
