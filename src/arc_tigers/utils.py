import torch
import yaml


def get_device() -> torch.device:
    """Gets the best available device for pytorch to use.
    (According to: gpu -> mps -> cpu) Currently only works for one GPU.

    Returns:
        torch.device: available torch device
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_yaml(yaml_file: str) -> dict:
    """Reads a yaml file and returns a dictionary.

    Args:
        yaml_file (str): path to the yaml file

    Returns:
        dict: dictionary with the contents of the yaml file
    """

    with open(yaml_file) as f:
        return yaml.safe_load(f)
