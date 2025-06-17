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


def get_distilbert_embeddings(
    dataset, eval_dir, embedding_savename="distilbert_embeddings"
):
    if os.path.isfile(eval_dir + f"{embedding_savename}.npy"):
        print(f"{embedding_savename}.npy Found in {eval_dir}")
        return np.load(eval_dir + f"{embedding_savename}.npy")

    print(f"{embedding_savename}.npy not found in {eval_dir}. Generating embeddings...")
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
    np.save(eval_dir + f"{embedding_savename}.npy", embeddings)

    return embeddings
