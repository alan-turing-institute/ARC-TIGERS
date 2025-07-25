from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from arc_tigers.constants import HPARAMS_CONFIG_DIR, OUTPUTS_DIR, TRAIN_CONFIG_DIR
from arc_tigers.data.config import HFDataConfig, SyntheticDataConfig, load_data_config
from arc_tigers.model.config import ModelConfig
from arc_tigers.utils import load_yaml


@dataclass
class HParamsConfig:
    config_name: str
    train_kwargs: dict[str, Any]

    @classmethod
    def from_name(cls, config_name: str) -> "HParamsConfig":
        """Load data config from a YAML file based on the config name."""
        config_path = HPARAMS_CONFIG_DIR / f"{config_name}.yaml"
        config = load_yaml(config_path)
        return cls(config_name=config_name, train_kwargs=config)


@dataclass
class TrainConfig:
    config_name: str
    model_config: ModelConfig
    data_config: HFDataConfig | SyntheticDataConfig
    hparams_config: HParamsConfig | None = None

    @classmethod
    def from_path(cls, config_path: str | Path) -> "TrainConfig":
        """Load data config from a YAML file."""
        config = load_yaml(config_path)
        data_config_name = config.pop("data_config")
        data_config = load_data_config(data_config_name)
        model_config_name = config.pop("model_config")
        model_config = ModelConfig.from_name(model_config_name)

        if (model_config.is_synthetic and isinstance(data_config, HFDataConfig)) or (
            not model_config.is_synthetic
            and isinstance(data_config, SyntheticDataConfig)
        ):
            msg = "Synthetic models and data can only be used together. "
            raise ValueError(msg)

        if not model_config.is_synthetic:
            hparams_config_name = config.pop("hparams_config")
            hparams_config = HParamsConfig.from_name(hparams_config_name)
        else:
            hparams_config = None

        return cls(
            config_name=Path(config_path).stem,
            data_config=data_config,
            model_config=model_config,
            hparams_config=hparams_config,
            **config,
        )

    @classmethod
    def from_name(cls, config_name: str) -> "TrainConfig":
        """Load data config from a YAML file based on the config name."""
        config_path = TRAIN_CONFIG_DIR / f"{config_name}.yaml"
        return cls.from_path(config_path)

    @property
    def model_dir(self) -> Path:
        path = (
            OUTPUTS_DIR
            / self.data_config.data_name
            / self.data_config.save_name
            / self.model_config.config_name
        )
        if self.hparams_config:
            path = path / self.hparams_config.config_name

        return path

    def save(self, path: str | Path | None = None):
        """Save the training configuration to a YAML file."""
        if path is None:
            path = self.model_dir
        if Path(path).is_dir():
            path = Path(path) / "train_config.yaml"

        config_dict = {
            "model_config": self.model_config.config_name,
            "data_config": self.data_config.config_name,
        }
        if self.hparams_config:
            config_dict["hparams_config"] = self.hparams_config.config_name

        with open(path, "w") as f:
            yaml.safe_dump(config_dict, f)

        return path
