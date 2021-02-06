import os
import re
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

def format_for_auto_phrase(all_text):
    text_lines = all_text.split("\n")
    formatted_text_lines = ["", ]
    for line in text_lines:
        if len(line):  # if the line actually contains text
            formatted_text_lines[-1] = " ".join(
                (
                    formatted_text_lines[-1],
                    line
                )
            )
        else: # otherwise, this is a paragraph break
            formatted_text_lines[-1] = formatted_text_lines[-1] + "." if formatted_text_lines[-1][-1] != "." else formatted_text_lines[-1]
            formatted_text_lines.append(".")
            formatted_text_lines.append("")

    formatted_text = "\n".join(formatted_text_lines)

    return formatted_text
