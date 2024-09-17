import argparse
import functools

import torch

from cortexchange.models.surf_stop_model import ImagenetTransferLearning, process_fits  # noqa
from cortexchange.predictor import Predictor

import __main__

setattr(__main__, "ImagenetTransferLearning", ImagenetTransferLearning)


class StopPredictor(Predictor):
    def __init__(self, model_name: str, device: str, *args, variational_dropout: int = 0, **kwargs):
        super().__init__(model_name, device)

        self.dtype = torch.float32

        self.model = self.model.to(self.dtype)
        self.model.eval()

        assert variational_dropout >= 0
        self.variational_dropout = variational_dropout

    @functools.lru_cache(maxsize=1)
    def prepare_data(self, input_path: str) -> torch.Tensor:
        input_data: torch.Tensor = torch.from_numpy(process_fits(input_path))
        input_data = input_data.to(self.dtype)
        input_data = input_data.swapdims(0, 2).unsqueeze(0)
        return input_data

    @torch.no_grad()
    def predict(self, data: torch.Tensor):
        with torch.autocast(dtype=self.dtype, device_type=self.device):
            if self.variational_dropout > 0:
                self.model.feature_extractor.eval()
                self.model.classifier.train()

            predictions = torch.concat(
                [torch.sigmoid(self.model(data)).clone() for _ in range(self.variational_dropout)],
                dim=1
            )

            mean = predictions.mean()
            std = predictions.std()

        print(mean, std)
        return mean, std

    @staticmethod
    def add_argparse_args(parser: argparse.ArgumentParser) -> None:
        parser.add_argument(
            "--variational_dropout",
            type=int,
            default=None,
            help="Optional: Amount of times to run the model to obtain a variational estimate of the stdev"
        )
