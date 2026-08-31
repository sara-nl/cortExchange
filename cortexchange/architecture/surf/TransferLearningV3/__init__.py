import argparse
import functools
import os

import torch
from astropy.io import fits
from torchvision.transforms.functional import normalize

from cortexchange.architecture import Architecture

from .inference import load_checkpoint
from .pre_processing import normalize_fits
from .resampling import resize_and_noise


def process_fits(fits_path):
    with fits.open(fits_path) as hdul:
        image_data = hdul[0].data

    return normalize_fits(image_data)


class TransferLearningV3(Architecture):
    """
    Same model family as TransferLearningV2, differing in how the input is prepared.

    Both versions load checkpoints through astroNNomy, so both apply the LoRA weights the
    checkpoint contains. What V3 changes is the preprocessing:

    - The dataset mean/std recorded in the checkpoint were looked up with getattr on a dict, which
      always missed, so inputs reached the model unnormalized. They are read with .get now.
    - Resampling injected Gaussian noise at prediction time, which made predictions stochastic.
      Resampling is antialiased and noise is opt-in through noise_sigma.

    Predictions therefore differ from V2 for the same weights.
    """

    def __init__(
        self,
        model_name: str = None,
        device: str = None,
        variational_dropout: int = 0,
        noise_sigma: float = 0.0,
        **kwargs,
    ):
        super().__init__(model_name, device)

        self.dtype = torch.bfloat16

        self.model = self.model.to(self.dtype)
        self.model.eval()

        assert variational_dropout >= 0
        self.variational_dropout = variational_dropout

        assert noise_sigma >= 0
        self.noise_sigma = noise_sigma

        self.resize = None

    def set_resize(self, resize: int) -> None:
        self.resize = resize

    def load_checkpoint(self, path) -> torch.nn.Module:
        # To avoid errors on CPU
        if "gpu" not in self.device and self.device != "cuda":
            os.environ["XFORMERS_DISABLED"] = "1"
        (
            model,
            self.optim,
            self.config,
        ) = load_checkpoint(path, self.device).values()

        return model

    @functools.lru_cache(maxsize=1)
    def prepare_data(self, input_path: str, **kwargs) -> torch.Tensor:
        input_data: torch.Tensor = torch.from_numpy(process_fits(input_path))
        input_data = input_data.to(self.dtype)
        input_data = input_data.swapdims(0, 2).unsqueeze(0)
        return self.prepare_batch(input_data, **kwargs)

    def prepare_batch(
        self, batch: torch.Tensor, mean=None, std=None, resize=None
    ) -> torch.Tensor:
        batch = batch.to(self.dtype).to(self.device)
        transforms = getattr(self.config, "data_transforms", {})

        if resize is None:
            if self.resize is not None:
                resize = self.resize
            else:
                resize = transforms.get("resize_val", resize)

        batch = self.resize_batch(batch, resize, self.noise_sigma)

        # data_transforms is a plain dict, so these have to be read with .get - getattr silently
        # missed and left the input unnormalized
        if mean is None:
            mean = transforms.get("mean", mean)

        if std is None:
            std = transforms.get("std", std)

        batch = self.normalize_batch(batch, mean, std)
        return batch

    @staticmethod
    def resize_batch(
        batch: torch.Tensor, resize: int, noise_sigma: float = 0.0
    ) -> torch.Tensor:
        if resize is not None:
            batch = resize_and_noise(batch, resize, noise_sigma)
        return batch

    @staticmethod
    def normalize_batch(
        batch: torch.Tensor,
        mean: torch.Tensor = None,
        std: torch.Tensor = None,
    ) -> torch.Tensor:
        if mean is None:
            mean = 0
        if std is None:
            std = 1
        return normalize(batch, mean=mean, std=std)

    @torch.no_grad()
    def predict(self, data: torch.Tensor):
        with torch.autocast(dtype=self.dtype, device_type=self.device):
            if self.variational_dropout > 0:
                self.model.train()
            else:
                self.model.eval()

            predictions = torch.concat(
                [
                    torch.sigmoid(self.model(data)).clone()
                    for _ in range(max(self.variational_dropout, 1))
                ],
                dim=1,
            )
            if self.variational_dropout > 0:
                mean = predictions.mean(dim=1)
                std = predictions.std(dim=1)
            else:
                mean = predictions[0]
                std = None

        return mean, std

    @staticmethod
    def add_argparse_args(parser: argparse.ArgumentParser) -> None:
        parser.add_argument(
            "--variational_dropout",
            type=int,
            default=0,
            help="Optional: Amount of times to run the model to obtain a variational estimate of the stdev",
        )
        parser.add_argument(
            "--noise_sigma",
            type=float,
            default=0.0,
            help="Optional: stdev of Gaussian noise added after resampling. 0 (the default) keeps "
            "inference deterministic; earlier versions always added noise.",
        )
