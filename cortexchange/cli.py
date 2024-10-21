import importlib
import logging
import os
import shutil
import sys
import traceback
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

from cortexchange.wdclient import init_downloader, client
from cortexchange.architecture import Architecture
from cortexchange.utils import create_argparse, create_argparse_upload, create_argparse_group, \
    create_argparse_upload_arch


def get_architecture_cls(architecture_type) -> type(Architecture):
    if architecture_type is None:
        raise ValueError(f"Please pass your model with `--model_architecture=group/model`.")

    segments = architecture_type.split("/", 1)
    if len(segments) == 1:
        raise ValueError(f"Invalid format: should be `--model_architecture=group/model`.")

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
        raise ValueError(f"Model {architecture_type} is not implemented in this version of cortExchange.")

    return architecture_cls


def run():
    args = create_argparse()
    init_downloader(url=args.wd_url, login=args.wd_login, password=args.wd_password, cache=args.cache)

    architecture_cls = get_architecture_cls(args.model_architecture)

    # Reinitialize args for specific predictor class.
    args = create_argparse(architecture_cls)

    predictor = architecture_cls(**vars(args))

    data = predictor.prepare_data(args.input)
    predictor.predict(data)


def upload_weights():
    args = create_argparse_upload()
    print(args.__dict__)
    if not os.path.exists(args.weights):
        raise ValueError(f"No such path exists: {args.weights}.")

    full_path_weights = os.path.abspath(args.weights)
    segments = full_path_weights.split("/")
    model_name = segments[-1].split(".")[0]
    temp_cache_path = "/".join(segments[:-2])

    if args.validate:
        print("Validation")
        init_downloader(url=args.wd_url, login=args.wd_login, password=args.wd_password, cache=temp_cache_path)

        architecture_cls = get_architecture_cls(args.model_architecture)
        args = create_argparse(architecture_cls)

        kwargs = vars(args)
        kwargs["model_name"] = model_name

        architecture = architecture_cls(**kwargs)

    # Take the args name, or use the filename if none is given.
    remote_model_name = args.weights_name if args.weights_name is not None else model_name
    init_downloader(url=args.wd_url, login=args.wd_login, password=args.wd_password, cache=args.cache)

    client.upload_model(remote_model_name, full_path_weights, force=args.force)


def upload_architecture():
    args = create_argparse_upload_arch()
    init_downloader(url=args.wd_url, login=args.wd_login, password=args.wd_password, cache=args.cache)

    architecture_name = args.architecture_name

    # First copy the architecture to cache to find out if it would work
    temp_arch_path = client.local_architecture_path(architecture_name)
    if os.path.exists(temp_arch_path) and not args.force:
        raise ValueError("This architecture already exists locally. Pass --force to overwrite.")

    print(args.architecture_root_path, client.local_architecture_path(architecture_name))

    shutil.copytree(args.architecture_root_path, temp_arch_path, dirs_exist_ok=True)
    try:
        arch_cls = get_architecture_cls(architecture_type=architecture_name)
        print(f"Model initialized correctly.")
        client.upload_architecture(args.architecture_name, args.architecture_root_path, force=args.force)
    except Exception as e:
        print(traceback.format_exc())
        pass  # Always remove temp files
    shutil.rmtree(temp_arch_path)

    # client.upload_architecture(args.architecture_name, args.architecture_root_path, force=args.force)


def create_group():
    args = create_argparse_group()
    init_downloader(url=args.wd_url, login=args.wd_login, password=args.wd_password, cache=args.cache)
    client.create_group(args.group_name)


def list_group():
    args = create_argparse_group()
    init_downloader(url=args.wd_url, login=args.wd_login, password=args.wd_password, cache=args.cache)

    print("\n## Architectures: ##")
    print("\n".join(f"\t{x}" for x in client.list_group(args.group_name, list_weights=False)))
    print("\n## Weights: ##")
    print("\n".join(f"\t{x}" for x in client.list_group(args.group_name)))


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
