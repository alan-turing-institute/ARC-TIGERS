import logging
import os
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
    pipeline,
)

from arc_tigers.data.config import HFDataConfig, SyntheticDataConfig
from arc_tigers.data.utils import tokenize_data
from arc_tigers.model.beta_model import BetaModel
from arc_tigers.samplers.acquisition import SurrogateData
from arc_tigers.sampling.metrics import compute_metrics
from arc_tigers.training.config import TrainConfig

logger = logging.getLogger(__name__)


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

    texts = dataset["text"]
    labels = dataset["label"]
    reddit_dataset = RedditTextDataset(texts, labels)
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
    elif train_config.model_config.model_id == "zero-shot":
        preds = get_zero_shot_preds(test_dataset, train_config)
    else:
        preds = get_transformers_preds(
            train_config, test_dataset, data_config.test_imbalance
        )

    os.makedirs(cache_path.parent, exist_ok=True)
    np.save(cache_path, preds)
    logger.info("Saved predictions to %s", cache_path)

    return preds


def get_zero_shot_preds(
    test_dataset: Dataset,
    train_config: TrainConfig,
) -> np.ndarray:
    """
    Get predictions from a zero-shot model using the transformers library.

    Args:
        test_dataset: Dataset to use for predictions.
        train_config: Train config, specifies the previously trained model to use.
        imbalance: Optional imbalance ratio for if train_config specifies a synthetic
            model.

    Returns:
        Predictions from the model on the test dataset.
    """

    # cast data_config to help the type checker
    data_config = train_config.data_config
    if not isinstance(data_config, HFDataConfig):
        err_msg = (
            "Zero-shot models are only supported with HFDataConfig, "
            "not SyntheticDataConfig."
        )
        raise ValueError(err_msg)

    model_path = "facebook/bart-large-mnli"
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    task = data_config.task

    if task == "one-vs-all":
        candidate_labels = ["something else", data_config.target_config]
    else:
        err_msg = f"Unsupported task: {task}. Expected 'one-vs-all'."
        raise ValueError(err_msg)

    model = pipeline(
        "zero-shot-classification",
        model=model_path,
        tokenizer=tokenizer,
        device=0 if torch.cuda.is_available() else -1,
    )

    logger.info("Computing zero-shot predictions...")
    outputs = model(
        test_dataset["text"],
        candidate_labels=candidate_labels,
        hypothesis_template=train_config.model_config.model_kwargs[
            "hypothesis_template"
        ],
    )

    # Extract numerical predictions properly with consistent ordering
    predictions = []
    for output in outputs:
        # Create a mapping from label to score for consistent ordering
        label_scores = dict(zip(output["labels"], output["scores"], strict=True))
        # Get scores for both classes in consistent order
        something_else_score = label_scores["something else"]
        target_score = label_scores[data_config.target_config]
        # Return as logits format [prob_class_0, prob_class_1]
        predictions.append([something_else_score, target_score])

    return np.array(predictions)


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
