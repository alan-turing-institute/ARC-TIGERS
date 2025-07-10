import json
import os
import random
from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml


def seed_everything(seed: int) -> None:
    """Set random seeds for torch, numpy, random, and python.

    Args:
        seed: Seed to set.
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


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


def load_yaml(yaml_file: str | Path) -> dict:
    """Reads a yaml file and returns a dictionary.

    Args:
        yaml_file (str): path to the yaml file

    Returns:
        dict: dictionary with the contents of the yaml file
    """

    with open(yaml_file) as f:
        return yaml.safe_load(f)


def array_to_list(obj: Any) -> Any:
    """Converts numpy arrays and torch tensors to lists, leaving other objects
    unchanged.

    Args:
        obj: Any python object, possibly containing numpy arrays or torch tensors.

    Returns:
        The input object with numpy arrays and torch tensors converted to lists.
    """
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, torch.Tensor):
        return obj.cpu().tolist()
    return obj


def to_json(obj: dict[str, Any] | list, file_path: str | Path) -> None:
    """Writes a python object to a json file, converting any numpy arrays or torch
    tensors to lists first.

    Args:
        obj: python Dict to write to file
        file_path: path to the file to write to
    """
    write_obj = deepcopy(obj)

    if isinstance(write_obj, list):
        write_obj = array_to_list(write_obj)
    else:
        for key, value in write_obj.items():
            write_obj[key] = array_to_list(value)

    with open(file_path, "w") as f:
        json.dump(write_obj, f)
