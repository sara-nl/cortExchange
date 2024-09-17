import functools
import os

import torch

from cortexchange.pre_processing_for_ml import process_fits
from cortexchange.predictor import Predictor
from cortexchange.models.surf_stop_model import ImagenetTransferLearning  # noqa
from cortexchange.models.surf_stop_model import load_checkpoint
from cortexchange.utils import download_model, create_argparse


class StopPredictor(Predictor):
    def __init__(self, cache, model_name: str, device: str, variational_dropout: int = 0):
        self.dtype = torch.float32
        self.device = device
        model_path = os.path.join(cache, model_name)
        if not os.path.exists(model_path):
            download_model(cache, model_name)

        checkpoint = load_checkpoint(model_path, device)
        super().__init__(checkpoint.get("model").to(self.dtype))

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
    def predict(self, input_path):
        data = self.prepare_data(input_path)

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
    def add_argparse_args(parser):
        parser.add_argument(
            "--variational_dropout",
            type=int,
            default=None,
            help="Optional: Amount of times to run the model to obtain a variational estimate of the stdev"
        )


def main(args):
    predictor = StopPredictor(
        cache=args.cache,
        device=args.device,
        model_name=args.model,
        variational_dropout=args.variational_dropout
    )
    print("Initialized models")
    predictor.predict(input_path=args.input)


if __name__ == "__main__":
    main(create_argparse())
