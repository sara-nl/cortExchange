import logging
import os
import sys
import tarfile

import webdav3
from tqdm import tqdm
from webdav3.client import Client
from webdav3.exceptions import RemoteResourceNotFound, ResponseErrorCode


class WDClient:
    options = {}
    cache = ""
    client = None
    bar = None

    ARCH_PREFIX = "architectures"
    WEIGHT_PREFIX = "weights"

    def initialize(self, url: str, login: str, password: str, cache: str):
        self.options = {
            "webdav_hostname": url,
            "webdav_login": login,
            "webdav_password": password,
        }
        self.cache = cache
        self.client = Client(self.options)
        try:
            print("Attempting authorization")
            self.client.list()
        except webdav3.exceptions.WebDavException as e:
            print(
                "Authorization failed with:\n\t"
                + f"\n\t".join(
                    f"{k}: {v if k != 'webdav_password' else '************'}"
                    for k, v in self.options.items()
                )
            )
            sys.exit()

    def local_weights_path(self, model_name: str):
        return os.path.join(self.cache, "weights", model_name)

    def local_architecture_path(self, architecture_name: str):
        return os.path.join(self.cache, "architectures", architecture_name)

    def remote_weights_path(self, model_name):
        return f"{self.WEIGHT_PREFIX}/{model_name}"

    def remote_architecture_path(self, architecture_name):
        return f"{self.ARCH_PREFIX}/{architecture_name}"

    def progress(self, current, total):
        if self.bar is None:
            self.bar = tqdm(total=total, unit_scale=True, unit="iB", unit_divisor=1024)
        self.bar.n = current
        self.bar.refresh()

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

        model_path = self.local_weights_path(model_name)
        if not force and os.path.exists(model_path):
            logging.info(f"Model already found at {model_path}")
            return

        tarred_file = f"{model_name}.tar.gz"
        full_path_tar = self.local_weights_path(tarred_file)

        files = self.client.list(self.remote_weights_path(group))
        if f"{model}.tar.gz" not in files:
            raise ValueError("No file exists remotely with this name.")

        os.makedirs(self.local_weights_path(group), exist_ok=True)
        self.bar = None
        self.client.download(
            self.remote_weights_path(tarred_file), full_path_tar, progress=self.progress
        )

        # Extract tar and remove original file.
        tar = tarfile.TarFile(full_path_tar)
        tar.extractall(self.local_weights_path(group))
        tar.close()
        os.remove(full_path_tar)

    def upload_model(self, model_name, weights_path, force=False):
        group, model = model_name.split("/")
        tarred_file = f"{model}.tar.gz"

        files = self.client.list(remote_path=self.remote_weights_path(group))
        if not force and tarred_file in files:
            raise ValueError("Model already exists remotely with this name.")

        full_path_tar = os.path.join(self.cache, tarred_file)

        # Create tar and upload
        tar = tarfile.TarFile(full_path_tar, mode="w")
        tar.add(weights_path, arcname=model)
        tar.close()

        self.bar = None
        try:
            self.client.upload(
                remote_path=self.remote_weights_path(f"{group}/{tarred_file}"),
                local_path=full_path_tar,
                progress=self.progress,
            )
        except ResponseErrorCode as e:
            if e.code == 403:
                raise ConnectionError(
                    "The given webdav credentials do not have write-rights."
                )
            else:
                raise e
        os.remove(full_path_tar)

    def upload_architecture(
        self, architecture_name, architecture_root_path, force=False
    ):
        group, architecture = architecture_name.split("/")
        tarred_file = f"{architecture}.tar.gz"

        files = self.client.list(remote_path=self.remote_architecture_path(group))
        if not force and tarred_file in files:
            raise ValueError("Architecture already exists remotely with this name.")

        full_path_tar = os.path.join(self.cache, tarred_file)

        # Create tar and upload
        tar = tarfile.TarFile(full_path_tar, mode="w")
        tar.add(architecture_root_path, arcname=architecture, recursive=True)
        tar.close()

        size_in_bytes = os.path.getsize(full_path_tar)
        mb_size_limit = 2
        if size_in_bytes > mb_size_limit * 1024 * 1024:  # 1MB max code size
            raise ValueError(
                f"Code size in directory exceeds {mb_size_limit}MB. "
                f"Please remove unnecessary (binary) files from the directory and try again."
            )
        else:
            self.bar = None
            try:
                self.client.upload(
                    remote_path=self.remote_architecture_path(f"{group}/{tarred_file}"),
                    local_path=full_path_tar,
                    progress=self.progress,
                )
            except ResponseErrorCode as e:
                if e.code == 403:
                    raise ConnectionError(
                        "The given webdav credentials do not have write-rights."
                    )
                else:
                    raise e

        os.remove(full_path_tar)

    def create_group(self, group_name: str):
        try:
            self.client.mkdir("weights")
            self.client.mkdir("architectures")
            self.client.mkdir(self.remote_weights_path(group_name))
            self.client.mkdir(self.remote_architecture_path(group_name))
        except ResponseErrorCode as e:
            if e.code == 403:
                raise ConnectionError(
                    "The given webdav credentials do not have write-rights."
                )
            else:
                raise e

    def list_group(self, group_name: str, list_weights=True):
        try:
            if not self.client.is_dir(self.remote_weights_path(group_name)):
                raise ValueError("Given group is not a group but a model.")
        except RemoteResourceNotFound:
            raise ValueError("No groups with this name exist.")

        if list_weights:
            return self.client.list(self.remote_weights_path(group_name))
        else:
            return self.client.list(self.remote_architecture_path(group_name))

    def download_architecture(self, architecture_name, force=False):
        if self.cache is None:
            raise RuntimeError(
                "Initialize downloader before accessing models: "
                "`from cortexchange.downloader import init_downloader; init_downloader(...)`"
            )

        group, model = architecture_name.split("/")

        architecture_path = self.local_architecture_path(architecture_name)
        if not force and os.path.exists(architecture_path):
            logging.info(f"Architecture already found at {architecture_path}")
            return

        tarred_file = f"{architecture_name}.tar.gz"
        local_path_tar = self.local_architecture_path(tarred_file)

        files = self.client.list(self.remote_architecture_path(group))
        if f"{model}.tar.gz" not in files:
            raise ValueError("No file exists remotely with this name.")

        os.makedirs(self.local_architecture_path(group), exist_ok=True)
        self.bar = None
        self.client.download(
            self.remote_architecture_path(tarred_file),
            local_path_tar,
            progress=self.progress,
        )

        # Extract tar and remove original file.
        tar = tarfile.TarFile(local_path_tar)
        tar.extractall(self.local_architecture_path(group))
        tar.close()
        os.remove(local_path_tar)


def init_downloader(url: str, login: str, password: str, cache: str):
    global client
    client.initialize(url, login, password, cache)


client = WDClient()


class DefaultWebdavArgs:
    URL = "https://researchdrive.surfsara.nl/public.php/webdav/"
    CACHE = f"{os.path.join(os.path.expanduser('~'), '.cache/cortexchange')}"
    LOGIN = os.getenv("WD_LOGIN", "WsSxVZHPqHlKcvY")
    PASSWORD = os.getenv("WD_PASSWORD", "PublicAccess1!")
