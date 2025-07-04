from dataclasses import dataclass
from pathlib import Path
from typing import Any

from arc_tigers.constants import TRAIN_CONFIG_DIR
from arc_tigers.data.config import DataConfig
from arc_tigers.model.config import ModelConfig
from arc_tigers.utils import load_yaml


@dataclass
class TrainConfig:
    config_name: str
    model_config: ModelConfig
    data_config: DataConfig
    exp_name: str
    random_seed: int
    train_kwargs: dict[str, Any]

    @classmethod
    def from_path(cls, config_path: str | Path) -> "TrainConfig":
        """Load data config from a YAML file."""
        config = load_yaml(config_path)
        data_config_name = config.pop("data_config")
        model_config_name = config.pop("model_config")
        return cls(
            config_name=Path(config_path).stem,
            data_config=DataConfig.from_name(data_config_name),
            model_config=ModelConfig.from_name(model_config_name),
            **config,
        )

    @classmethod
    def from_name(cls, config_name: str) -> "TrainConfig":
        """Load data config from a YAML file based on the config name."""
        config_path = TRAIN_CONFIG_DIR / f"{config_name}.yaml"
        return cls.from_path(config_path)
