import os

import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


class RedditTextDataset(Dataset):
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


def get_distilbert_embeddings(
    dataset, storage_dir, embedding_savename="distilbert_embeddings"
):
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
    model = SentenceTransformer("sentence-transformers/distilbert-base-nli-mean-tokens")

    all_embeddings = []
    all_labels = []

    for text, labels in tqdm(dataloader):
        embeddings_batch = model.encode(text, convert_to_tensor=True)

        all_embeddings.append(embeddings_batch)
        all_labels.append(labels)

    embeddings = torch.vstack(all_embeddings).detach().cpu().numpy()
    np.save(storage_dir + f"{embedding_savename}.npy", embeddings)

    return embeddings
