from pathlib import Path

PROJECT_DIR = Path.cwd()
CONFIG_DIR = Path(PROJECT_DIR, "configs")
DATA_DIR = Path(PROJECT_DIR, "data")
OUTPUTS_DIR = Path(PROJECT_DIR, "outputs")


def _get_config_dir(location):
    """Helper function for creating project config paths.

    Args:
        location: Directory inside PROJECT_ROOT/configs/ to create path for

    Returns:
        String giving path of PROJECT_ROOT/configs/location
    """
    return CONFIG_DIR / location


MODEL_CONFIG_DIR = _get_config_dir("model")
DATA_CONFIG_DIR = _get_config_dir("data")
TRAIN_CONFIG_DIR = _get_config_dir("training")
TASKS_CONFIG_DIR = _get_config_dir("tasks")
HPARAMS_CONFIG_DIR = _get_config_dir("hparams")
