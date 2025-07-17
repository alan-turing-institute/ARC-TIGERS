import argparse
import logging

from arc_tigers.training.config import TrainConfig
from arc_tigers.training.train import train_transformers

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(description="Train a transformers classifier")
    parser.add_argument("train_config", help="path to the train config yaml file")
    args = parser.parse_args()
    train_config = TrainConfig.from_path(args.train_config)
    train_transformers(train_config)
