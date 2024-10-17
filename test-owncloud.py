import base64
import os
from typing import Optional

import owncloud
import dotenv
from owncloud import ShareInfo

import random
import string


def get_project_id(root_dirs):
    while True:
        try:
            project_id = int(input('Project ID: '))
            if 0 <= project_id < len(root_dirs):
                return project_id
            else:
                print('Invalid project ID. Please choose a valid option.')
        except ValueError:
            print('Invalid input. Please enter a valid project ID.')


import string
import random


def generate_random_password(length):
    special = "!@#$%^&*()"
    characters = string.ascii_letters + string.digits + special
    password = ''.join(random.choice(characters) for _ in range(length))
    return password

class OCClient:
    """
    A class representing an OwnCloud client.

    Attributes:
    oc (owncloud.Client): The OwnCloud client instance.
    project_dir (str): The path to the project directory.
    """
    def __init__(self, project_dir: Optional[str] = None):
        self.oc = owncloud.Client('https://researchdrive.surfsara.nl/')
        self.oc.login(os.getenv("OC_LOGIN"), os.getenv("OC_PASSWORD"))
        self.project_dir = project_dir or self.get_project_dir_input()

    def path(self, *args: str) -> str:
        return "/".join([self.project_dir, *args])

    def get_project_dir_input(self) -> str:
        root_dirs = self.oc.list("/")
        print('Which of the following directories is used for cortExchange.')
        print('\n'.join(f'{i}: {x.path}' for i, x in enumerate(root_dirs)))
        project_id = get_project_id(root_dirs)
        return root_dirs[project_id].path

    def revoke_access_tokens(self, group_name):
        shares: list[ShareInfo] = self.oc.get_shares(self.path(group_name))
        response = input(f"Do you want to delete {len(shares)} share urls? (y/n): ")
        if response.lower() == 'y':
            for share in shares:
                self.oc.delete_share(share.share_id)

    def create_access_token(self, group_name):
        password = generate_random_password(20)
        share_weights: ShareInfo = self.oc.share_file_with_link(
            self.path("weights", group_name),
            public_upload=True,
            password=password,
        )

        share_architectures: ShareInfo = self.oc.share_file_with_link(
            self.path("architectures", group_name),
            public_upload=True,
            password=password
        )

        token = base64.b64encode(
            f"w={share_weights.token};a={share_architectures.token};p={password}".encode()
        ).decode()
        accept = input(
            f"The following token is a base64 plaintext encoding of the password and access link. Whomever this token is "
            f"shared with will have write access to the `{group_name}` directories. Continue with this token? (y/n)"
        ).lower()

        if accept == 'y':
            print(token)
        else:
            self.oc.delete_share(share_weights.share_id)
            self.oc.delete_share(share_architectures.share_id)
            print("No access links are created.")


def main():
    client = OCClient()
    client.create_access_token("surf")


if __name__ == '__main__':
    dotenv.load_dotenv()
    main()
