import argparse
import os

from cortexchange.architecture import Architecture


def create_argparse(architecture_cls: type(Architecture) = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Default arguments for cortExchange models.", add_help=architecture_cls is not None
    )
    parser.add_argument(
        "--model_architecture",
        type=str,
        help="What model configuration to use. This is formatted as 'group/model_name'."
    )

    parser.add_argument(
        "--cache",
        type=str,
        default=f"{os.path.join(os.path.expanduser('~'), '.cache/cortexchange')}",
        help="Where to store the downloaded model weights."
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="surf/version_7743995_4__model_resnext101_64x4d__lr_0.001__normalize_0__dropout_p_0.25__use_compile_1",
        help="Name of the model."
    )
    parser.add_argument("--input", type=str, default=None, help="Path to the input file.")
    parser.add_argument("--device", type=str, default="cpu", help="Device for inference, default=cpu.")

    _add_wd_args(parser)

    if architecture_cls is not None:
        architecture_cls.add_argparse_args(parser)
        return parser.parse_args()
    else:
        return parser.parse_known_args()[0]


def create_argparse_upload() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Arguments for uploading new cortExchange models."
    )
    parser.add_argument(
        "--weights",
        type=str,
        help="Full path to weights, should be a *.pth file."
    )
    parser.add_argument(
        "--validate",
        default=True,
        type=bool,
        help="Validate the model by loading it to memory before uploading."
    )
    parser.add_argument(
        "--force",
        default=False,
        type=bool,
        help="Force overwrite any remote models with the same name."
    )
    parser.add_argument(
        "--model_architecture",
        type=str,
        help="The Predictor class to load."
    )
    parser.add_argument(
        "--model_name",
        type=str,
        help="Name of the model. In format group/model_name. Leaving this emtpy will use the filename instead."
    )

    _add_wd_args(parser)
    return parser.parse_args()


def create_argparse_upload_arch() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Arguments for uploading new cortExchange models."
    )
    parser.add_argument(
        "--architecture_name",
        type=str,
        help="Name for the architecture."
    )
    parser.add_argument(
        "--architecture_root_path",
        default=True,
        type=bool,
        help="Full path to the root of the architecture code."
    )
    parser.add_argument(
        "--force",
        default=False,
        type=bool,
        help="Force overwrite any remote models with the same name."
    )

    _add_wd_args(parser)
    return parser.parse_args()


def create_argparse_group() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Arguments for creating new cortExchange groups."
    )
    parser.add_argument("--group_name", type=str, help="Group name for which to create the group.")

    _add_wd_args(parser)
    return parser.parse_args()


def _add_wd_args(parser):
    """
    Common webdav arguments.

    :param parser:
    :return:
    """
    parser.add_argument(
        "--wd-url",
        type=str,
        default="https://surfdrive.surf.nl/files/public.php/webdav/",
        help="URL where webdav is available for the to-be-downloaded models."
    )
    parser.add_argument(
        "--wd-login",
        type=str,
        default="5lnKaoagQi92y0j",
        help="Name of the directory in which the models are stored in webdav."
    )
    parser.add_argument("--wd-password", type=str, default="1234", help="Password for the webdav storage.")
