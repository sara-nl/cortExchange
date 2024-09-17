import importlib
import logging
import sys

from cortexchange.downloader import init_downloader
from cortexchange.predictor import Predictor
from cortexchange.utils import create_argparse


def run(args):
    model_type: str = args.model_configuration

    if model_type is None:
        logging.error(f"Please pass your model with `--model_configuration=organization/model`.")
        return exit(1)

    segments = model_type.split("/", 1)
    if len(segments) == 1:
        logging.error(f"Invalid format: should be `--model_configuration=organization/model`.")
        return exit(1)

    org, name = segments
    try:
        module_org = importlib.import_module(f"cortexchange.predictor.{org}")
        predictor_cls = getattr(module_org, name)
    except (ImportError, AttributeError):
        logging.error(
            f"No module found with name {model_type}. "
            f"Pass a valid predictor module with `--model_configuration=organization/model`."
        )
        return exit(1)

    if not isinstance(predictor_cls, type(Predictor)):
        logging.error(f"Model {model_type} is not implemented in this version of cortExchange.")
        return exit(1)

    # Reinitialize args for specific predictor class.
    args = create_argparse(predictor_cls)

    init_downloader(url=args.wd_url, login=args.wd_login, password=args.wd_password, cache=args.cache)
    predictor = predictor_cls(**vars(args))

    data = predictor.prepare_data(args.input)
    predictor.predict(data)


def main():
    args = create_argparse()
    methods = {
        "run": run
    }
    if len(sys.argv) < 2:
        logging.error(f"First argument mut be one of: {', '.join(methods.keys())}")
        return exit(1)

    cmd = sys.argv.pop(1)
    methods[cmd](args)


if __name__ == "__main__":
    main()
