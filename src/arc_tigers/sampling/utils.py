import logging
import os
from copy import deepcopy
from glob import glob
from pathlib import Path

import joblib
import numpy as np
import torch
from datasets import Dataset
from datasets import Dataset as HFDataset
from sentence_transformers import SentenceTransformer
from sklearn.pipeline import Pipeline
from torch.utils.data import DataLoader
from torch.utils.data import Dataset as TorchDataset
from tqdm import tqdm
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)

from arc_tigers.data.config import HFDataConfig, SyntheticDataConfig
from arc_tigers.data.utils import tokenize_data
from arc_tigers.model.beta_model import BetaModel
from arc_tigers.samplers.acquisition import SurrogateData
from arc_tigers.samplers.sampler import Sampler
from arc_tigers.sampling.bias import BiasCorrector
from arc_tigers.sampling.metrics import compute_metrics, evaluate
from arc_tigers.training.config import TrainConfig

logger = logging.getLogger(__name__)


def sample_dataset_metrics(
    dataset: Dataset,
    preds: np.ndarray,
    sampler: Sampler,
    evaluate_steps: list[int],
    bias_corrector: BiasCorrector | None = None,
) -> list[dict[str, float]]:
    """
    Simulate iteratively random sampling the whole dataset, re-computing metrics
    after each sample.

    Args:
        dataset: The dataset to sample from.
        preds: The predictions for the dataset.
        model: The model to compute metrics with.
        seed: The random seed for sampling.
        max_labels: The maximum number of labels to sample. If None, the whole dataset
            will be sampled.

    Returns:
        A list of dictionaries containing the computed metrics after each sample.
    """
    max_labels = evaluate_steps[-1]
    evaluate_steps = deepcopy(evaluate_steps)
    metrics = []
    next_eval_step = evaluate_steps.pop(0)
    for n in tqdm(range(max_labels)):
        _, q = sampler.sample()
        if bias_corrector is not None:
            bias_corrector.compute_weighting_factor(q_im=q, m=n + 1)
        if (n + 1) == next_eval_step:
            metric = evaluate(
                dataset[sampler.labelled_idx],
                preds[sampler.labelled_idx],
                bias_corrector=bias_corrector,
            )
            metric["n"] = n + 1
            metrics.append(metric)
            if evaluate_steps:
                next_eval_step = evaluate_steps.pop(0)
            else:
                break

    return metrics


class RedditTextDataset(TorchDataset):
    def __init__(self, texts, labels):
        self.texts = texts
        self.labels = labels

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        return self.texts[idx], self.labels[idx]


def get_embeddings(
    dataset: HFDataset,
    cache_path: Path,
    model_id: str = "sentence-transformers/all-mpnet-base-v2",
) -> np.ndarray:
    """Get embeddings for a dataset using a pre-trained SentenceTransformer model. If
    embeddings are already cached, load them from the cache. Otherwise, generate and
    cache the embeddings.

    Args:
        dataset: The dataset containing text and labels.
        cache_path: Path to save/load the embeddings.
        model_id: Identifier for the SentenceTransformer model to use.

    Returns:
        Numpy array of embeddings.
    """
    if os.path.isfile(cache_path):
        logger.info("Found embeddings in %s", cache_path)
        return np.load(cache_path)

    logger.info("Generating embeddings with %s...", model_id)
    model = SentenceTransformer(model_id)

    train_texts = dataset["text"]
    train_labels = dataset["label"]
    reddit_dataset = RedditTextDataset(train_texts, train_labels)
    dataloader = DataLoader(reddit_dataset, batch_size=32, shuffle=True)

    all_embeddings = []
    for text, _ in tqdm(dataloader):
        embeddings_batch = model.encode(text, convert_to_tensor=True)
        all_embeddings.append(embeddings_batch)

    embeddings = torch.vstack(all_embeddings).detach().cpu().numpy()
    os.makedirs(cache_path.parent, exist_ok=True)
    np.save(cache_path, embeddings)
    logger.info("Embeddings saved to %s", cache_path)

    return embeddings


def get_preds_cache_path(
    train_config: TrainConfig,
    split: str,
    data_config: HFDataConfig | SyntheticDataConfig | None = None,
) -> Path:
    """Directory for saving cached predictions.

    Args:
        train_config: Configuration for the training run which specifies where the model
            is saved.
        split: Data split for which the cache is created, either "train" or "test"
        data_config: Configuration for the test dataset, required if split is "test".

    Returns:
        Path to the cache directory, a sub-directory of 'cache' in train_config.save_dir
    """
    path = train_config.model_dir / f"cache/{split}"
    if split == "test":
        if data_config is None:
            msg = "data_config must be provided for test split."
            raise ValueError(msg)
        te_imb = str(data_config.test_imbalance).replace(".", "")
        path = path / te_imb
    elif split != "train":
        msg = f"Invalid split: {split}. Expected 'train' or 'test'."
        raise ValueError(msg)

    return path / "predictions" / "predictions.npy"


def get_surrogate_data(
    data_config: HFDataConfig, split: str, dataset: HFDataset | None = None
) -> SurrogateData:
    """Get surrogate data (embeddings and labels) for train or test splits, usually
    for the purposes of a surrogate model or other proxy using embeddings.

    Args:
        data_config: Configuration for the dataset.
        split: Either "train" or "test".
        dataset: Optional pre-loaded dataset to use instead of loading from config.

    Returns:
        Dictionary with keys "embeds" and "labels".
    """
    if split not in ["train", "test"]:
        msg = f"Invalid split: {split}. Expected 'train' or 'test'."
        raise ValueError(msg)
    if dataset is None:
        if split == "train":
            dataset = data_config.get_train_split()
        else:
            dataset = data_config.get_test_split()

    cache_dir = data_config.test_dir if split == "test" else data_config.full_splits_dir
    cache_dir = cache_dir / "cache"
    embeds_cache_path = cache_dir / "embed" / "mpnet" / "embeds.npy"

    embeds = get_embeddings(dataset, embeds_cache_path)
    labels = np.array(dataset["label"])

    return SurrogateData(embeds, labels, cache_dir)


def get_eval_outputs_dir(
    train_config: TrainConfig,
    data_config: HFDataConfig | SyntheticDataConfig,
    acq_strat: str,
) -> Path:
    return (
        train_config.model_dir
        / "eval_outputs"
        / str(data_config.test_imbalance).replace(".", "")
        / acq_strat
    )


def get_preds(
    train_config: TrainConfig,
    data_config: HFDataConfig | SyntheticDataConfig,
    test_dataset: Dataset | None = None,
) -> np.ndarray:
    """
    Wrapper function to get predictions from either a transformer model or a tfidf
    model.
    """
    cache_path = get_preds_cache_path(train_config, "test", data_config)
    preds_exist = os.path.exists(cache_path)
    if preds_exist:
        logger.info("Loading cached predictions from %s", cache_path)
        return np.load(cache_path)

    if test_dataset is None:
        test_dataset = data_config.get_test_split()

    if len(glob(f"{train_config.model_dir}/*.joblib")) > 0:
        preds = get_tfidf_preds(train_config, test_dataset)
    preds = get_transformers_preds(
        train_config, test_dataset, data_config.test_imbalance
    )

    os.makedirs(cache_path.parent, exist_ok=True)
    np.save(cache_path, preds)
    logger.info("Saved predictions to %s", cache_path)

    return preds


def get_transformers_preds(
    train_config: TrainConfig, test_dataset: Dataset, imbalance: float | None = None
) -> np.ndarray:
    """
    Get the predictions from a model using the transformers library. This function is
    also used when synthetic data is used.

    Args:
        train_config: Train config, specifies the previously trained model to use
        data_config: Data config, specifies the test dataset to use for predictions.
        test_dataset: Optional pre-loaded test dataset from data config.
        imbalance: Optional imbalance ratio for if train_config specifies a synthetic
            model

    Returns:
        predictions from the model on the test dataset
    """
    model_path = train_config.model_dir
    if train_config.model_config.is_synthetic:
        # Arbitrary tokenizer only loaded for compatibility with other functions, not
        # used by the synthetic Beta model.
        logger.info("Loading roberta-base tokenizer as placeholder for synthetic model")
        tokenizer = AutoTokenizer.from_pretrained("roberta-base")
        use_cpu = True
    else:
        # calculate predictions for whole dataset
        logger.info("Loading tokenizer from %s...", model_path)
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        use_cpu = False

    test_dataset = tokenize_data(test_dataset, tokenizer)

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    if train_config.model_config.is_synthetic:
        if imbalance is None:
            msg = "Test imbalance must be set for synthetic models."
            raise ValueError(msg)

        model_kwargs = train_config.model_config.model_kwargs
        pos_err_rate = model_kwargs["positive_error_rate"]
        neg_err_rate = model_kwargs["negative_error_rate"]
        if pos_err_rate or neg_err_rate:
            if not (pos_err_rate and neg_err_rate):
                msg = "Both positive and negative error rates must be set."
                raise ValueError(msg)
            model = BetaModel.from_error_rates(pos_err_rate, neg_err_rate)
        else:
            model = BetaModel.from_imbalance_and_advantage(
                imbalance, model_kwargs["model_adv"]
            )
    else:
        logger.info("Loading model from %s...", model_path)
        model = AutoModelForSequenceClassification.from_pretrained(model_path)

    training_args = TrainingArguments(
        output_dir="tmp", per_device_eval_batch_size=16, use_cpu=use_cpu
    )
    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        processing_class=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )
    logger.info("Computing predictions...")
    return trainer.predict(test_dataset, metric_key_prefix="").predictions


def get_tfidf_preds(train_config: TrainConfig, test_dataset: Dataset) -> np.ndarray:
    logger.info("Loading model from %s...", train_config.model_dir)
    model_path = glob(f"{train_config.model_dir}/*.joblib")[0]
    model: Pipeline = joblib.load(model_path)

    return model.predict(test_dataset["text"])
