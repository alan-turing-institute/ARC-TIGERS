import json
import os
import random
from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

from arc_tigers.constants import DATA_CONFIG_DIR, MODEL_CONFIG_DIR


def config_path_to_config_name(config_path: str) -> str:
    return config_path.split("/")[-1].rstrip(".yaml")


def create_dirs(
    save_dir: str, data_config_path: str, class_balance: float, acq_strat: str
) -> tuple[str, str, str]:
    data_config = config_path_to_config_name(data_config_path)
    eval_dir = f"{save_dir}/eval_outputs/{data_config}/"
    if class_balance != 1.0:
        output_dir = (
            f"{eval_dir}/imbalanced_{acq_strat}_sampling_outputs_"
            f"{str(class_balance).replace('.', '')}/"
        )
        predictions_dir = (
            f"{save_dir}/eval_outputs/data_cache/{data_config}/predictions/"
            f"imbalanced_{str(class_balance).replace('.', '')}/"
        )
        embeddings_dir = (
            f"{save_dir}/eval_outputs/data_cache/{data_config}/embeddings/"
            f"imbalanced_{str(class_balance).replace('.', '')}/"
        )
    else:
        output_dir = f"{eval_dir}/{acq_strat}_sampling_outputs/"
        predictions_dir = (
            f"{save_dir}/eval_outputs/data_cache/{data_config}/predictions/balanced/"
        )
        embeddings_dir = (
            f"{save_dir}/eval_outputs/data_cache/{data_config}/embeddings/balanced/"
        )
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(predictions_dir, exist_ok=True)
    os.makedirs(embeddings_dir, exist_ok=True)

    return output_dir, predictions_dir, embeddings_dir


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


def to_json(obj: dict[str, Any] | list, file_path: str) -> None:
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
