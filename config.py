import os

def get_data_path():
    running_in_ci = os.environ.get("CI", "false") == "true"

    if running_in_ci:
        return "data/raw/"
    else:
        return "../data/raw/"

def get_output_path():
    return "../outputs/"
