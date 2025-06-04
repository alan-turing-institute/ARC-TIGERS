from typing import Any
import numpy as np
import random
import os
import torch
import yaml

from arc_tigers.constants import DATA_CONFIG_DIR, MODEL_CONFIG_DIR


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


def load_yaml(yaml_file: str) -> dict:
    """Reads a yaml file and returns a dictionary.

    Args:
        yaml_file (str): path to the yaml file

    Returns:
        dict: dictionary with the contents of the yaml file
    """

    with open(yaml_file) as f:
        return yaml.safe_load(f)


def get_configs(exp_config: dict[str, Any]):
    """Get the experiment, data and model configs from the experiment config file.

    Args:
        exp_config (str): path to the experiment config file

    Returns:
        tuple: experiment config, data config, model config
    """
    data_config_file_name = f"{exp_config['data_config']}.yaml"
    model_config_file_name = f"{exp_config['model_config']}.yaml"
    data_config = load_yaml(DATA_CONFIG_DIR / data_config_file_name)
    model_config = load_yaml(MODEL_CONFIG_DIR / model_config_file_name)
    return data_config, model_config
