import importlib
import logging
import os
import sys

from cortexchange.wdclient import init_downloader, client
from cortexchange.predictor import Predictor
from cortexchange.utils import create_argparse, create_argparse_upload, create_argparse_group


def get_predictor_cls(model_type) -> type(Predictor):
    if model_type is None:
        logging.error(f"Please pass your model with `--model_configuration=group/model`.")
        return exit(1)

    segments = model_type.split("/", 1)
    if len(segments) == 1:
        logging.error(f"Invalid format: should be `--model_configuration=group/model`.")
        return exit(1)

    org, name = segments
    try:
        module_org = importlib.import_module(f"cortexchange.predictor.{org}")
        predictor_cls = getattr(module_org, name)
    except (ImportError, AttributeError):
        logging.error(
            f"No module found with name {model_type}. "
            f"Pass a valid predictor module with `--model_configuration=group/model`."
        )
        return exit(1)

    if not isinstance(predictor_cls, type(Predictor)):
        logging.error(f"Model {model_type} is not implemented in this version of cortExchange.")
        return exit(1)

    return predictor_cls


def run():
    args = create_argparse()

    predictor_cls = get_predictor_cls(args.model_configuration)

    # Reinitialize args for specific predictor class.
    args = create_argparse(predictor_cls)

    init_downloader(url=args.wd_url, login=args.wd_login, password=args.wd_password, cache=args.cache)
    predictor = predictor_cls(**vars(args))

    data = predictor.prepare_data(args.input)
    predictor.predict(data)


def upload():
    args = create_argparse_upload()

    if not os.path.exists(args.weights):
        logging.error(f"No such path exists: {args.weights}.")
        return exit(1)

    full_path_weights = os.path.abspath(args.weights)
    segments = full_path_weights.split("/")
    model_name = segments[-1].split(".")[0]
    temp_cache_path = "/".join(segments[:-2])

    if args.validate:
        init_downloader(url=args.wd_url, login=args.wd_login, password=args.wd_password, cache=temp_cache_path)

        predictor_cls = get_predictor_cls(args.model_configuration)
        args = create_argparse(predictor_cls)

        kwargs = vars(args)
        kwargs["model_name"] = model_name

        predictor = predictor_cls(**kwargs)

    client.upload_model(model_name, full_path_weights, force=args.force)


def create_group():
    args = create_argparse_group()
    client.create_group(args.group_name)


def main():
    methods = {
        "run": run,
        "upload": upload,
        "create-group": create_group
    }
    if len(sys.argv) < 2:
        logging.error(f"First argument mut be one of: {', '.join(methods.keys())}")
        return exit(1)

    cmd = sys.argv.pop(1)
    methods[cmd]()


if __name__ == "__main__":
    main()
