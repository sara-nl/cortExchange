import importlib
import logging

from cortexchange.downloader import init_downloader
from cortexchange.predictor import Predictor
from cortexchange.utils import create_argparse


def main():
    args = create_argparse()

    model_type: str = args.model_configuration

    if model_type is None:
        logging.error(f"Please pass your model with `--model_configuration=organization/model`.")

    segments = model_type.split("/", 1)
    if len(segments) == 1:
        logging.error(f"Invalid format: should be `--model_configuration=organization/model`.")
        return exit(1)

    org, name = segments
    try:
        module_org = importlib.import_module(f"cortexchange.predictor.{org}")
        predictor = getattr(module_org, name)
    except (ImportError, AttributeError):
        logging.error(
            f"No module found with name {model_type}. "
            f"Pass a valid predictor module with `--model_configuration=organization/model`."
        )
        return exit(1)

    if not isinstance(predictor, type(Predictor)):
        logging.error(f"Model {model_type} is not implemented in this version of cortExchange.")
        return exit(1)

    # Reinitialize args for specific predictor class.
    args = create_argparse(predictor)

    init_downloader(*vars(args))

    predictor = predictor(
        device=args.device,
        model_name=args.model,
        variational_dropout=args.variational_dropout
    )

    predictor.predict(input_path=args.input)


if __name__ == "__main__":
    main()
