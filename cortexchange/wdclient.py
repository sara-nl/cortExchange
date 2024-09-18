import logging
import os
import tarfile

from tqdm import tqdm
from webdav3.client import Client


class WDClient:
    def __init__(self, url: str, login: str, password: str, cache: str):
        self.options = {
            'webdav_hostname': url,
            'webdav_login': login,
            'webdav_password': password
        }
        self.cache = cache
        self.client = Client(self.options)
        self.bar = None

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

        group, model = model_name.split("/")

        model_path = self.get_path(model)
        if not force and os.path.exists(model_path):
            logging.info(f"Model already found at {model_path}")
            return

        tarred_file = f'{model}.tar.gz'
        full_path_tar = os.path.join(self.cache, tarred_file)

        files = self.client.list(group)
        if tarred_file not in files:
            logging.error(f"Available files: {files}")
            raise ValueError("No file exists remotely with this name.")

        os.makedirs(self.cache, exist_ok=True)
        self.bar = None
        self.client.download(tarred_file, full_path_tar, progress=self.progress)

        # Extract tar and remove original file.
        tar = tarfile.TarFile(full_path_tar)
        tar.extractall(self.cache)
        tar.close()
        os.remove(full_path_tar)

    def progress(self, current, total):
        if self.bar is None:
            self.bar = tqdm(total=total, unit_scale=True, unit="B", unit_divisor=1024)
        self.bar.n = current
        self.bar.refresh()

    def upload_model(self, model_name, weights_path, force=False):
        group, model = model_name.split("/")
        tarred_file = f'{model}.tar.gz'

        files = self.client.list(remote_path=group)
        if not force and tarred_file in files:
            logging.error(f"Available files: {files}")
            raise ValueError("File already exists remotely with this name.")

        full_path_tar = os.path.join(self.cache, tarred_file)

        # Create tar and upload
        tar = tarfile.TarFile(full_path_tar)
        tar.add(weights_path)
        tar.close()

        self.bar = None
        self.client.upload(f"{group}/{tarred_file}", full_path_tar, progress=self.progress)
        os.remove(full_path_tar)

    def create_group(self, group_name: str):
        if self.client.is_dir(group_name):
            raise ValueError("Group already exists with this name.")
        self.client.mkdir(group_name)


def init_downloader(
    url: str = "https://surfdrive.surf.nl/files/public.php/webdav/",
    login: str = "5lnKaoagQi92y0j",
    password: str = "1234",
    cache: str = os.path.expanduser("~/.cache/cortexchange")
):
    global client
    client = WDClient(url, login, password, cache)


client: WDClient
init_downloader()
