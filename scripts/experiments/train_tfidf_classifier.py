import argparse

from arc_tigers.training.config import TrainConfig
from arc_tigers.training.train import train_tfidf

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a TF-IDF classifier")
    parser.add_argument("train_config", help="path to the train config yaml file")
    args = parser.parse_args()
    train_config = TrainConfig.from_path(args.train_config)
    train_tfidf(train_config)
