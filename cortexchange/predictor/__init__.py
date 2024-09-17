import abc
from typing import Any

import torch


class Predictor(abc.ABC):
    def __init__(self, model: torch.nn.Module):
        self.model = model

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
