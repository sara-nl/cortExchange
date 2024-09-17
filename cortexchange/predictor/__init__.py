import abc
from typing import Any

import torch

from cortexchange.downloader import downloader
from cortexchange.models.surf.stop_model import load_checkpoint


class Predictor(abc.ABC):
    model: torch.nn.Module

    def __init__(self, model_name, device, *args, **kwargs):
        assert model_name is not None, "Pass --model_name in the arguments list."
        assert device is not None, "Pass --device in the arguments list."

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
    def predict(self, data: Any) -> Any:
        pass

    @staticmethod
    def add_argparse_args(parser):
        """
        Add additional arguments specific to the predictor implementation.

        :param parser:
        :return:
        """
        pass
