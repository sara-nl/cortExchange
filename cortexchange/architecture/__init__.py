import abc
import importlib
import logging
import os
import sys
import traceback
from pathlib import Path
from typing import Any, Optional
from dotenv import load_dotenv

load_dotenv()

import torch

from cortexchange.wdclient import client, DefaultWebdavArgs

# Init webdav client if not set by user specifically
if client.client is None:

    client.initialize(
        DefaultWebdavArgs.URL,
        DefaultWebdavArgs.LOGIN,
        DefaultWebdavArgs.PASSWORD,
        DefaultWebdavArgs.CACHE,
    )


class Architecture(abc.ABC):
    model: torch.nn.Module

    def __init__(self, model_name, device, *args, **kwargs):
        assert model_name is not None, "Pass --model_name in the arguments list."
        assert device is not None, "Pass --device in the arguments list."

        self.device = device
        client.download_model(model_name)
        self.model = self.load_checkpoint(client.local_weights_path(model_name))

    @abc.abstractmethod
    def load_checkpoint(self, path):
        pass

    @abc.abstractmethod
    def prepare_data(self, data: Any) -> torch.Tensor:
        """
        This method needs to be implemented by child classes.
        Converts any input data format to tensors that are processable by the model.
        :return:
        """
        ...

    @torch.no_grad()
    def predict(self, data: torch.Tensor) -> Any:
        pass

    @staticmethod
    def add_argparse_args(parser):
        """
        Add additional arguments specific to the predictor implementation.

        :param parser:
        :return:
        """
        pass


def get_architecture(architecture_type) -> type(Architecture):
    if architecture_type is None:
        raise ValueError(
            f"Please pass your model with `--model_architecture=group/model`."
        )

    segments = architecture_type.split("/", 1)
    if len(segments) == 1:
        raise ValueError(
            f"Invalid format: should be `--model_architecture=group/model`."
        )

    org, name = segments
    os.makedirs(client.local_architecture_path(f"{org}"), exist_ok=True)

    def try_import(module_name) -> Optional[type(Architecture)]:
        try:
            # Attempt importing base-package model
            module_org = importlib.import_module(module_name)
            return getattr(module_org, name)
        except (ImportError, AttributeError) as e:
            logging.warning(f"Error while importing {module_name}")
            traceback.print_exc()
            return None

    architecture_cls = try_import(f"cortexchange.architecture.{org}.{name}")
    if architecture_cls is None:
        # Add cache to path and create init file for module import
        sys.path.append(client.local_architecture_path(""))
        Path(client.local_architecture_path(f"{org}/__init__.py")).touch()

        architecture_cls = try_import(f"{org}.{name}")
    if architecture_cls is None:
        client.download_architecture(architecture_type)
        architecture_cls = try_import(f"{org}.{name}")

    if architecture_cls is None:
        raise ValueError(
            f"No module found with name {architecture_type}. "
            f"Pass a valid predictor module with `--model_architecture=group/model`."
        )

    if not isinstance(architecture_cls, type(Architecture)):
        raise ValueError(
            f"Model {architecture_type} is not implemented in this version of cortExchange."
        )

    return architecture_cls
