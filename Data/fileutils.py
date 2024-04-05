import os
from typing import Sequence


# utils to work with files

def get_sub_folders(root_folder):
    return [os.path.join(dirpath, dirname)
            for dirpath, dirnames, _ in os.walk(root_folder)
            for dirname in dirnames]


def get_files(folder_path: str, file_extension: str = '') -> Sequence[str]:
    return filter(lambda file: file.endswith(file_extension), os.listdir(folder_path))
