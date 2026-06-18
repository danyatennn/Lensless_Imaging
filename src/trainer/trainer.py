import torch

from src.metrics.tracker import MetricTracker
from src.trainer.base_trainer import BaseTrainer
from src.utils.lensless import crop_roi


class Trainer(BaseTrainer):
    def process_batch(self, batch, metrics: MetricTracker):
        batch = self.move_batch_to_device(batch)
        batch = self.transform_batch(batch)

        metric_funcs = self.metrics["inference"]
        if self.is_train:
            metric_funcs = self.metrics["train"]
            self.optimizer.zero_grad()

        outputs = self.model(**batch)
        batch.update(outputs)

        all_losses = self.criterion(**batch)
        batch.update(all_losses)

        if self.is_train:
            batch["loss"].backward()
            self._clip_grad_norm()
            self.optimizer.step()
            if self.lr_scheduler is not None:
                self.lr_scheduler.step()

        for loss_name in self.config.writer.loss_names:
            metrics.update(loss_name, batch[loss_name].item())

        for met in metric_funcs:
            metrics.update(met.name, met(**batch))
        return batch

    def _log_batch(self, batch_idx, batch, mode="train"):
        if mode != "train":
            self.log_predictions(batch)

    def log_predictions(self, batch, n_examples=4):
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
