import os
from typing import Generator


def files_in_folder(folder_full_path:str) -> Generator[str, None, None]:
    for file_name in os.listdir(folder_full_path):
        yield folder_full_path + os.sep + file_name


def get_last_file_in_folder(folder_full_path:str) -> str:
    file_names = [file for file in os.listdir(folder_full_path)]
    return file_names[-1]