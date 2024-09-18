import importlib
import logging
import os
import sys
from dotenv import load_dotenv

from cortexchange.wdclient import init_downloader, client
from cortexchange.architecture import Architecture
from cortexchange.utils import create_argparse, create_argparse_upload, create_argparse_group, \
    create_argparse_upload_arch


def get_architecture_cls(architecture_type) -> type(Architecture):
    if architecture_type is None:
        logging.error(f"Please pass your model with `--model_architecture=group/model`.")
        return exit(1)

    segments = architecture_type.split("/", 1)
    if len(segments) == 1:
        logging.error(f"Invalid format: should be `--model_architecture=group/model`.")
        return exit(1)

    org, name = segments
    try:
        module_org = importlib.import_module(f"cortexchange.architecture.{org}.{name}")
        predictor_cls = getattr(module_org, name)
    except (ImportError, AttributeError):
        client.download_architecture(architecture_type)

        logging.error(
            f"No module found with name {architecture_type}. "
            f"Pass a valid predictor module with `--model_architecture=group/model`."
        )
        return exit(1)

    if not isinstance(predictor_cls, type(Architecture)):
        logging.error(f"Model {architecture_type} is not implemented in this version of cortExchange.")
        return exit(1)

    return predictor_cls


def run():
    args = create_argparse()

    architecture_cls = get_architecture_cls(args.model_architecture)

    # Reinitialize args for specific predictor class.
    args = create_argparse(architecture_cls)

    init_downloader(url=args.wd_url, login=args.wd_login, password=args.wd_password, cache=args.cache)
    predictor = architecture_cls(**vars(args))

    data = predictor.prepare_data(args.input)
    predictor.predict(data)


def upload_weights():
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

        architecture_cls = get_architecture_cls(args.model_architecture)
        args = create_argparse(architecture_cls)

        kwargs = vars(args)
        kwargs["model_name"] = model_name

        architecture = architecture_cls(**kwargs)

    # Take the args name, or use the filename if none is given.
    remote_model_name = args.model_name if args.model_name is not None else model_name

    client.upload_model(remote_model_name, full_path_weights, force=args.force)


def upload_architecture():
    args = create_argparse_upload_arch()
    client.upload_architecture(args.architecture_name, args.architecture_root_path, force=args.force)


def create_group():
    args = create_argparse_group()
    client.create_group(args.group_name)


def list_group():
    args = create_argparse_group()
    print("\n".join(client.list_group(args.group_name)))


def main():
    methods = {
        "run": run,
        "upload-weights": upload_weights,
        "upload-architecture": upload_architecture,
        "create-group": create_group,
        "list-group": list_group,
    }
    if len(sys.argv) < 2 or sys.argv[1] not in methods.keys():
        logging.error(f"First argument must be one of: {', '.join(methods.keys())}")
        return exit(1)

    cmd = sys.argv.pop(1)
    methods[cmd]()


if __name__ == "__main__":
    load_dotenv()
    main()
