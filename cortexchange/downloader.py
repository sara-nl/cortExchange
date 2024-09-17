import logging
import os
import tarfile

from tqdm import tqdm
from webdav3.client import Client


class Downloader:
    def __init__(self, url: str, login: str, password: str, cache: str):
        self.options = {
            'webdav_hostname': url,
            'webdav_login': login,
            'webdav_password': password
        }
        self.cache = cache

    def get_path(self, model_name: str):
        return os.path.join(self.cache, model_name)

    def download_model(self, model_name: str, force=False):
        """
        Downloads a model from SURFDrive or ResearchDrive including the code to allow for local importing.

        :param model_name:
        :param force: force redownloading a model even if it already exists.
        :return:
        """
        if self.cache is None:
            raise RuntimeError(
                "Initialize downloader before accessing models: "
                "`from cortexchange.downloader import init_downloader; init_downloader(...)`"
            )

        model_path = self.get_path(model_name)
        if not force and os.path.exists(model_path):
            logging.info(f"Model already found at {model_path}")
            return

        tarred_file = f'{model_name}.tar.gz'
        full_path_tar = os.path.join(self.cache, tarred_file)

        client = Client(self.options)
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

        os.makedirs(self.cache, exist_ok=True)
        client.download(tarred_file, full_path_tar, progress=progress)

        # Extract tar and remove original file.
        tar = tarfile.TarFile(full_path_tar)
        tar.extractall(self.cache)
        tar.close()
        os.remove(full_path_tar)


def init_downloader(
    url: str = "https://surfdrive.surf.nl/files/public.php/webdav/",
    login: str = "5lnKaoagQi92y0j",
    password: str = "1234",
    cache: str = ".cache"
):
    global downloader
    downloader = Downloader(url, login, password, cache)


downloader: Downloader
init_downloader()
