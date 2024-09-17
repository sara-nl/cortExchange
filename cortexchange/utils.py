import argparse

from cortexchange.predictor import Predictor


def create_argparse(predictor_class: type(Predictor) = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Default arguments for cortExchange models.", add_help=predictor_class is not None
    )
    parser.add_argument(
        "--model_configuration",
        type=str,
        help="What model configuration to use. This is formatted as 'organization/model_name'."
    )

    parser.add_argument("--cache", type=str, default="~/.cache/cortexchange", help="Where to store the downloaded model weights.")
    parser.add_argument(
        "--model",
        type=str,
        default="version_7743995_4__model_resnext101_64x4d__lr_0.001__normalize_0__dropout_p_0.25__use_compile_1",
        help="Name of the model."
    )
    parser.add_argument("--input", type=str, default=None, help="Path to the fits file.")
    parser.add_argument("--device", type=str, default="cpu", help="Device for inference, default=cpu.")

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

    if predictor_class is not None:
        predictor_class.add_argparse_args(parser)
        return parser.parse_args()
    else:
        return parser.parse_known_args()[0]
