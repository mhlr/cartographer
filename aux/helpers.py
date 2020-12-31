import os
from shutil import rmtree


def create_output_dir(path, delete_if_exists=False):

    # ignore if it already exists, but it's empty
    if os.path.isdir(path) and len(os.listdir(path)) == 0:
        return

    # create if it doesn't already exist
    elif not os.path.isdir(path):
        os.makedirs(path)
        return

    # directory already exists, but it's not empty
    else:
        if delete_if_exists:
            rmtree(path)
            create_output_dir(path)
        else:
            raise RuntimeError(f"Directory {path} already exists")
