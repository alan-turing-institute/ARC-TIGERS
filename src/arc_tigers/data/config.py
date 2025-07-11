import os
from dataclasses import dataclass
from pathlib import Path

from datasets import Dataset, DatasetDict

from arc_tigers.constants import DATA_CONFIG_DIR, DATA_DIR, TASKS_CONFIG_DIR
from arc_tigers.data.synthetic import get_synthetic_data
from arc_tigers.utils import load_yaml


@dataclass
class SyntheticDataConfig:
    config_name: str
    config_path: Path
    data_name: str
    n_rows: int
    test_imbalance: float
    seed: int

    @classmethod
    def from_path(cls, config_path: str | Path) -> "SyntheticDataConfig":
        """Load data config from a YAML file."""
        config = load_yaml(config_path)
        return cls(
            config_name=Path(config_path).stem, config_path=Path(config_path), **config
        )

    @classmethod
    def from_name(cls, config_name: str) -> "SyntheticDataConfig":
        """Load data config from a YAML file based on the config name."""
        config_path = DATA_CONFIG_DIR / f"{config_name}.yaml"
        return cls.from_path(config_path)

    def get_test_split(self) -> Dataset:
        return get_synthetic_data(
            self.n_rows, imbalance=self.test_imbalance, seed=self.seed
        )

    @property
    def save_name(self) -> str:
        imb = str(self.test_imbalance).replace(".", "")
        return f"{imb}/{self.seed}_{self.n_rows}"

    @property
    def test_dir(self) -> Path:
        return DATA_DIR / self.data_name / self.save_name


@dataclass
class HFDataConfig:
    config_name: str
    config_path: Path
    data_name: str
    task: str
    target_config: str
    train_imbalance: float | None
    test_imbalance: float | None
    seed: int
    max_train_targets: int | None
    max_test_targets: int | None

    @classmethod
    def from_path(cls, config_path: str | Path) -> "HFDataConfig":
        """Load data config from a YAML file."""
        config = load_yaml(config_path)
        return cls(
            config_name=Path(config_path).stem, config_path=Path(config_path), **config
        )

    @classmethod
    def from_name(cls, config_name: str) -> "HFDataConfig":
        """Load data config from a YAML file based on the config name."""
        config_path = DATA_CONFIG_DIR / f"{config_name}.yaml"
        return cls.from_path(config_path)

    @property
    def parent_data_dir(self) -> Path:
        """Path to the full parent dataset"""
        return DATA_DIR / self.data_name

    @property
    def save_name(self) -> str:
        tr_imb = str(self.train_imbalance).replace(".", "")
        return f"{self.task}/{self.target_config}/{self.seed}_{tr_imb}"

    @property
    def splits_dir(self) -> Path:
        return self.parent_data_dir / f"splits/{self.save_name}/"

    @property
    def full_splits_dir(self) -> Path:
        """Path to the full train and test splits for this config. This contains the
        full training set for this config, and the full test set from which the actual
        test split for this config is derived (by taking a subset to achieve the
        required level of imbalance)."""
        return self.splits_dir / "full"

    @property
    def test_dir(self) -> Path:
        """Path to the test split for this config."""
        te_imb = str(self.test_imbalance).replace(".", "")
        return self.splits_dir / f"test/{te_imb}"

    @property
    def target_categories(self) -> dict[str, list[str]]:
        """Target categories (sub-reddits) for the train and test splits."""
        if self.task == "one-vs-all":
            one_vs_all_config = load_yaml(TASKS_CONFIG_DIR / "one_vs_all.yaml")
            return one_vs_all_config[self.target_config]
        if self.task == "binary":
            binary_config = load_yaml(TASKS_CONFIG_DIR / "binary.yaml")
            return binary_config[self.target_config]

        drift_config = load_yaml(TASKS_CONFIG_DIR / "drift.yaml")
        return drift_config[self.target_config]

    def get_parent_data(self) -> Dataset:
        return Dataset.load_from_disk(self.parent_data_dir)

    def get_full_splits(self) -> DatasetDict:
        return DatasetDict.load_from_disk(self.full_splits_dir)

    def get_train_split(self) -> Dataset:
        return self.get_full_splits()["train"]

    def get_test_split(self) -> Dataset:
        return Dataset.load_from_disk(self.test_dir)


def load_data_config(config_name_or_path: str) -> HFDataConfig | SyntheticDataConfig:
    if not os.path.exists(config_name_or_path):
        config_path = DATA_CONFIG_DIR / f"{config_name_or_path}.yaml"
    else:
        config_path = config_name_or_path

    config_type = load_yaml(config_path)["data_name"]
    if config_type == "synthetic":
        return SyntheticDataConfig.from_path(config_path)

    return HFDataConfig.from_path(config_path)
