import abc
from typing import Any

import torch

from cortexchange.downloader import downloader
from cortexchange.models.surf_stop_model import load_checkpoint


class Predictor(abc.ABC):
    model: torch.nn.Module

    def __init__(self, model_name, device):
        self.device = device
        downloader.download_model(model_name)
        checkpoint = load_checkpoint(downloader.get_path(model_name), self.device)

        self.model = checkpoint.get("model")

    @abc.abstractmethod
    def prepare_data(self, data: Any) -> torch.Tensor:
        """
        This method needs to be implemented by child classes.
        Converts any input data format to tensors that are processable by the model.
        :return:
        """
        ...

    @torch.no_grad()
    def predict(self):
        pass

    @staticmethod
    def add_argparse_args(parser):
        """
        Add additional arguments specific to the predictor implementation.

        :param parser:
        :return:
        """
        pass
