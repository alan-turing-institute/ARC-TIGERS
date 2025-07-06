import os
from pathlib import Path

import numpy as np
import torch
from datasets import Dataset as HFDataset
from sentence_transformers import SentenceTransformer
from torch.utils.data import DataLoader
from torch.utils.data import Dataset as TorchDataset
from tqdm import tqdm

from arc_tigers.data.config import HFDataConfig, SyntheticDataConfig
from arc_tigers.training.config import TrainConfig


class RedditTextDataset(TorchDataset):
    def __init__(self, texts, labels):
        self.texts = texts
        self.labels = labels

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        return self.texts[idx], self.labels[idx]


def non_linear_spacing(min_labels, max_labels, n_steps=None):
    """
    Returns a non-linearly spaced list of evaluation steps:
    - More granular between 10 and 50 (step 5)
    - Steps of 50 until 300
    - Steps of 100 until 1000
    - Steps of 1000 until max_labels
    If n_steps is given and less than the number of steps, subsample to n_steps.
    """
    steps = []
    # Steps of 5 until 20
    for i in range(max(min_labels, 5), min(20, max_labels) + 1, 5):
        steps.append(i)
    # Steps of 10 until 50
    for i in range(max(20, min_labels + 1), min(50, max_labels) + 1, 10):
        steps.append(i)
    # 50 to 300 (step 50)
    for i in range(max(50, min_labels + 1), min(300, max_labels) + 1, 50):
        steps.append(i)
    # 300 to 1000 (step 100)
    for i in range(max(300, min_labels + 1), min(1000, max_labels) + 1, 100):
        steps.append(i)
    # 1000 to max_labels (step 1000)
    for i in range(max(1000, min_labels + 1), max_labels + 1, 1000):
        steps.append(i)
    # Ensure unique and sorted
    steps = sorted({s for s in steps if s <= max_labels})
    # Subsample if n_steps is set and less than total steps
    if n_steps is not None and n_steps < len(steps):
        idxs = np.linspace(0, len(steps) - 1, n_steps, dtype=int)
        steps = [steps[i] for i in idxs]
    return steps


def get_embeddings(
    dataset: HFDataset,
    storage_dir: str,
    embedding_savename: str,
    model: SentenceTransformer,
) -> np.ndarray:
    if os.path.isfile(storage_dir + f"{embedding_savename}.npy"):
        print(f"{embedding_savename}.npy Found in {storage_dir}")
        return np.load(storage_dir + f"{embedding_savename}.npy")

    print(
        f"{embedding_savename}.npy not found in {storage_dir}. Generating embeddings..."
    )
    train_texts = dataset["text"]
    train_labels = dataset["label"]

    reddit_dataset = RedditTextDataset(train_texts, train_labels)
    dataloader = DataLoader(reddit_dataset, batch_size=32, shuffle=True)

    all_embeddings = []
    all_labels = []

    for text, labels in tqdm(dataloader):
        embeddings_batch = model.encode(text, convert_to_tensor=True)

        all_embeddings.append(embeddings_batch)
        all_labels.append(labels)

    embeddings = torch.vstack(all_embeddings).detach().cpu().numpy()
    np.save(storage_dir + f"{embedding_savename}.npy", embeddings)

    return embeddings


def get_distilbert_embeddings(
    dataset: HFDataset,
    storage_dir: str,
    embedding_savename: str = "distilbert_embeddings",
) -> np.ndarray:
    model = SentenceTransformer("sentence-transformers/distilbert-base-nli-mean-tokens")

    return get_embeddings(dataset, storage_dir, embedding_savename, model)


def get_mpnet_embeddings(
    dataset: HFDataset, storage_dir: str, embedding_savename: str = "mpnet_embeddings"
) -> np.ndarray:
    model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")

    return get_embeddings(dataset, storage_dir, embedding_savename, model)


def get_preds_cache_path(
    train_config: TrainConfig,
    split: str,
    data_config: HFDataConfig | SyntheticDataConfig | None = None,
) -> Path:
    """Directory for saving cached predictions or other data.

    Args:
        train_config: Configuration for the training run which specifies where the model
            is saved.
        split: Data split for which the cache is created, either "train" or "test"
        data_config: Configuration for the test dataset, required if split is "test".

    Returns:
        Path to the cache directory, a sub-directory of 'cache' in train_config.save_dir
    """
    path = train_config.save_dir / f"cache/{split}"
    if split == "test":
        if data_config is None:
            msg = "data_config must be provided for test split."
            raise ValueError(msg)
        te_imb = str(data_config.test_imbalance).replace(".", "")
        path = path / te_imb
    elif split != "train":
        msg = f"Invalid split: {split}. Expected 'train' or 'test'."
        raise ValueError(msg)

    return path / "predictions/predictions.npy"
