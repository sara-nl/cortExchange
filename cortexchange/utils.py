import argparse
import logging
import os
import tarfile

from tqdm import tqdm
from webdav3.client import Client


def download_model(cache: str, model_name: str):
    """
    Downloads a model from SURFDrive or ResearchDrive including the code to allow for local importing.

    :param cache:
    :param model_name:
    :return:
    """
    tarred_file = f'{model_name}.tar.gz'
    full_path_tar = os.path.join(cache, tarred_file)

    options = {
        'webdav_hostname': "https://surfdrive.surf.nl/files/public.php/webdav/",
        'webdav_login': "5lnKaoagQi92y0j",
        'webdav_password': "1234"
    }

    client = Client(options)
    files = client.list()
    if tarred_file not in files:
        logging.error(f"Available files: {files}")
        raise ValueError("No file exists remotely with this name.")

    bar = None

    def progress(current, total):
        nonlocal bar
        if bar is None:
            bar = tqdm(total=total, unit_scale=True, unit="B", unit_divisor=1024)
        bar.n = current
        bar.refresh()

    os.makedirs(cache, exist_ok=True)
    client.download(tarred_file, full_path_tar, progress=progress)

    # Extract tar and remove original file.
    tar = tarfile.TarFile(full_path_tar)
    tar.extractall(cache)
    tar.close()
    os.remove(full_path_tar)


def create_argparse():
    parser = argparse.ArgumentParser(
        description="Default arguments for cortExchange models."
    )

    parser.add_argument("--cache", type=str, default=".cache", help="Where to store the downloaded model weights.")
    parser.add_argument(
        "--model",
        type=str,
        default="version_7743995_4__model_resnext101_64x4d__lr_0.001__normalize_0__dropout_p_0.25__use_compile_1",
        help="Name of the model."
    )
    parser.add_argument("--input", type=str, default=None, help="Path to the fits file.")
    parser.add_argument("--device", type=str, default="cpu", help="Device for inference, default=cpu.")
    return parser
