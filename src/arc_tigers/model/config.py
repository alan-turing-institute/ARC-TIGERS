from dataclasses import dataclass
from pathlib import Path
from typing import Any

from arc_tigers.constants import MODEL_CONFIG_DIR
from arc_tigers.utils import load_yaml


@dataclass
class ModelConfig:
    config_name: str
    model_id: str
    model_kwargs: dict[str, Any]

    @classmethod
    def from_path(cls, config_path: str | Path) -> "ModelConfig":
        """Load data config from a YAML file."""
        config = load_yaml(config_path)
        return cls(config_name=Path(config_path).stem, **config)

    @classmethod
    def from_name(cls, config_name: str) -> "ModelConfig":
        """Load data config from a YAML file based on the config name."""
        config_path = MODEL_CONFIG_DIR / f"{config_name}.yaml"
        return cls.from_path(config_path)

    @property
    def is_synthetic(self) -> bool:
        return self.model_id == "beta_model"
