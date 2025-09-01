import argparse
import functools

import torch
from torchvision.transforms.functional import normalize
import os

from cortexchange.architecture import Architecture

import __main__
from astropy.io import fits
from .utils import load_checkpoint, resize_and_noise
from .pre_processing_for_ml import normalize_fits


def process_fits(fits_path):
    with fits.open(fits_path) as hdul:
        image_data = hdul[0].data

    return normalize_fits(image_data)


class TransferLearningV2(Architecture):
    def __init__(
        self,
        model_name: str = None,
        device: str = None,
        variational_dropout: int = 0,
        **kwargs,
    ):
        try:
            super().__init__(model_name, device)
        except ModuleNotFoundError as e:
            if "No module named 'astronnomy'" in str(e):
                raise ImportError("It seems that the astronnomy module is not installed. Install with `pip install git+https://github.com/LOFAR-VLBI/astroNNomy.git#egg=astroNNomy`")
            raise e

        self.dtype = torch.bfloat16

        self.model = self.model.to(self.dtype)
        self.model.eval()

        assert variational_dropout >= 0
        self.variational_dropout = variational_dropout

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
        if resize is None:
            if self.resize is not None:
                resize = self.resize
            else:
                resize = getattr(self.config, "data_transforms", {}).get(
                    "resize_val", resize
                )

        batch = self.resize_batch(batch, resize)

        if mean is None:
            mean = getattr(self.config.data_transforms, "mean", mean)

        if std is None:
            std = getattr(self.config.data_transforms, "std", std)

        batch = self.normalize_batch(batch, mean, std)
        return batch

    @staticmethod
    def resize_batch(batch: torch.Tensor, resize: int) -> torch.Tensor:
        if resize is not None:
            batch = resize_and_noise(batch, resize)
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
