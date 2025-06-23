import os
from collections import namedtuple

import joblib
import numpy as np
from datasets import Dataset
from scipy.spatial.distance import cdist
from sklearn.ensemble import IsolationForest
from sklearn.linear_model import SGDClassifier

from arc_tigers.eval.utils import softmax
from arc_tigers.sample.utils import get_distilbert_embeddings

DummyAcqFunc = namedtuple("DummyAcqFunc", ["observed_idx", "remaining_idx", "rng"])


class AcquisitionFunction:
    """
    Abstract base class for active learning acquisition functions.

    Provides common properties and methods for managing labelled and unlabelled indices,
    as well as sampling from a probability mass function (pmf) over the unlabelled pool.

    Attributes:
        remaining_idx (list[int]): Indices of unlabelled samples.
        observed_idx (list[int]): Indices of labelled samples.
        rng (np.random.BitGenerator): Random number generator for sampling.

    Methods:
        sample_pmf(pmf): Samples an index from the unlabelled pool according to the pmf.
        sample(): Abstract method to select the next sample to label.
    """

    def __init__(self):
        self.remaining_idx: list[int] = []
        self.observed_idx: list[int] = []
        self.rng: np.random.BitGenerator = None
        self.name: str = "AcquisitionFunction"

    @property
    def labelled_idx(self) -> list[int]:
        """
        Returns:
            Indices of the samples that have been labelled.
        """
        return self.observed_idx

    @property
    def unlabelled_idx(self) -> list[int]:
        """
        Returns:
            Indices of the samples that haven't been labelled.
        """
        return self.remaining_idx

    def sample_pmf(self, pmf):
        pmf = np.asarray(pmf, dtype=np.float64)  # Use higher precision

        pmf = np.clip(pmf, 0, 1)  # Ensure values are in [0, 1]
        total = pmf.sum()
        if not np.isfinite(total) or total == 0:
            err_msg = f"Invalid pmf: sum is {total}"
            raise ValueError(err_msg)
        pmf = pmf / total  # Renormalize
        if np.any(np.isnan(pmf)) or np.any(pmf < 0) or np.any(pmf > 1):
            err_msg = "pmf contains NaNs or values outside [0, 1]"
            raise ValueError(err_msg)

        prob = self.rng.multinomial(1, pmf)
        idx = np.where(prob)[0][0]
        sample_prob = pmf[idx]
        test_idx = self.remaining_idx[idx]
        self.observed_idx.append(int(test_idx))
        self.remaining_idx.remove(int(test_idx))

        return sample_prob

    def sample(self) -> int:
        raise NotImplementedError


class DistanceSampler(AcquisitionFunction):
    """
    Acquisition function that selects samples based on their average distance
    (in embedding space) to the currently labelled set.

    At each step, computes the mean Euclidean distance from each unlabelled sample
    to all labelled samples, and samples according to a softmax over these distances.

    Args:
        data (Dataset): The dataset to sample from.
        sampling_seed (int): Random seed for reproducibility.
        eval_dir (str): Directory for evaluation outputs.

    Methods:
        sample(): Selects the next sample to label based on distance.
    """

    def __init__(self, data: dict[str, Dataset], sampling_seed: int, eval_dir: str):
        """Samples a dataset based on distance between observed and unobserved
        embeddings.

        Args:
            data: The dataset to sample from.

        Properties:
            n_sampled: Number of samples that have been labelled so far.
            labelled_idx: Indices of the samples that have been labelled.
            unlabelled_idx: Indices of the samples that haven't been labelled.
        """
        self.eval_data = data["evaluation_dataset"]

        self.observed_idx: list[int] = []
        self.remaining_idx: list[int] = np.arange(len(self.eval_data)).tolist()
        self.n_sampled = 0
        self.rng = np.random.default_rng(seed=sampling_seed)

        # Get embeddings for whole dataset
        self.embeddings = get_distilbert_embeddings(self.eval_data, eval_dir)
        self.name = "distance"

    def sample(self) -> int:
        """
        Updates the n_sampled counter and gets the next sample to label.

        Returns:
            Index of the next sample to be labelled
        """

        # If no samples have been selected yet, sample from a uniform distribution
        if len(self.observed_idx) == 0:
            N = len(self.eval_data)
            # float16 causes an underflow error for large datasets
            pmf = np.ones(N, dtype=np.float64) / N

        else:
            # Subset embeddings
            remaining = self.embeddings[np.array(self.remaining_idx)]
            observed = self.embeddings[np.array(self.observed_idx)]

            # # Broadcasting to get all paired differences
            # d = remaining[:, np.newaxis, :] - observed
            # d = d**2
            # # Sum over feature dimension
            # d = d.sum(-1)
            # # sqrt to get distance
            # d = np.sqrt(d)
            # # Mean over other pairs
            # distances = d.mean(1)

            # Compute all pairwise Euclidean distances
            dists = cdist(remaining, observed, metric="euclidean")
            # Mean distance to observed for each remaining sample
            distances = dists.mean(axis=1)

            # Constract PDF via softmax
            pmf = np.exp(distances)
            pmf /= pmf.sum()

        return self.sample_pmf(pmf)


class AccSampler(AcquisitionFunction):
    """
    Acquisition function that selects samples based on disagreement between
    a surrogate Random Forest classifier and the main model.

    At each step, computes the expected loss (1 - surrogate accuracy) for each
    unlabelled sample, and samples according to a normalized probability over these
    losses.

    Args:
        data (dict[str, Dataset]): Dictionary containing datasets.
        eval_dir (str): Directory for evaluation outputs.
        sampling_seed (int): Random seed for reproducibility.

    Methods:
        set_model_preds(preds): Sets the model's predicted probabilities.
        surrogate_pretrain(): Pre-trains the surrogate model if not already trained.
        update_surrogate(): Updates surrogate predictions based on labelled data.
        accuracy_loss(): Computes expected loss for unlabelled samples.
        sample(): Selects the next sample to label based on expected loss.
    """

    def __init__(self, data: dict[str, Dataset], eval_dir: str, sampling_seed: int):
        """Samples a dataset based on distance between model loss and surrogate loss,
        where the surrogate is a Random Forest classifier.

        Args:
            data: The dataset to sample from.

        Properties:
            n_sampled: Number of samples that have been labelled so far.
            labelled_idx: Indices of the samples that have been labelled.
            unlabelled_idx: Indices of the samples that haven't been labelled.
        """
        self.n_sampled = 0
        self.model_preds = np.array([])
        self.surrogate_preds = np.array([])
        self.eval_data = data["evaluation_dataset"]
        self.surrogate_train_data = data["surrogate_train_dataset"]
        self.observed_idx: list[int] = []
        self.remaining_idx = np.arange(len(self.eval_data)).tolist()
        self.eval_dir = eval_dir

        # Get embeddings for use with RF classifier
        self.embeddings = get_distilbert_embeddings(self.eval_data, eval_dir)
        self.rng = np.random.default_rng(seed=sampling_seed)

        # initialise surrogate predictions and model
        self.num_classes = len(np.unique(self.eval_data["label"]))
        self.surrogate_preds = np.full(
            (len(self.eval_data), self.num_classes), 1.0 / self.num_classes
        )
        self.surrogate_model = SGDClassifier(
            loss="log_loss", random_state=self.rng.integers(1e9)
        )

        # Placeholder for model predictions (should be set externally)
        self.model_preds = np.zeros_like(self.surrogate_preds)
        self.name = "acc_sampler"

    def set_model_preds(self, preds: np.ndarray):
        """Set model predictions externally (shape: [n_samples, n_classes])."""
        self.model_preds = softmax(preds)

    def surrogate_pretrain(self):
        X_train = get_distilbert_embeddings(
            self.surrogate_train_data,
            self.eval_dir,
            embedding_savename="surrogate_embeddings",
        )
        y_train = np.array(self.surrogate_train_data["label"])
        surrogate_path = os.path.join(
            self.eval_dir,
            f"pretrained_surrogate_{self.surrogate_model.__class__.__name__}.joblib",
        )
        if os.path.exists(surrogate_path):
            print(f"Loading pre-trained surrogate model from {surrogate_path} ...")
            self.surrogate_model = joblib.load(surrogate_path)
        else:
            print(f"Pre-training surrogate model and saving to {surrogate_path} ...")
            self.surrogate_model = self.surrogate_model.fit(X_train, y_train)
            joblib.dump(self.surrogate_model, surrogate_path)

    def update_surrogate(self):
        if len(self.observed_idx) == 0:
            self.surrogate_preds = np.zeros_like(self.surrogate_preds)
            return
        X_acquired = self.embeddings[np.array(self.observed_idx)]
        y_acquired = np.array(self.eval_data["label"])[np.array(self.observed_idx)]

        self.surrogate_model = self.surrogate_model.partial_fit(X_acquired, y_acquired)
        # Get surrogate predictions for all samples
        intermediate_surrogate_preds = self.surrogate_model.predict_proba(
            self.embeddings
        )
        classes_ = self.surrogate_model.classes_
        # Fill a full array with zeros for all classes
        full_proba = np.zeros((len(self.eval_data), self.num_classes))
        # fill the full array with surrogate predictions
        # for each class in the surrogate predictions
        for i, c in enumerate(classes_):
            full_proba[:, c] = intermediate_surrogate_preds[:, i]
        self.surrogate_preds = full_proba

    def accuracy_loss(self):
        # Use only remaining (unlabelled) samples
        # we need higher values = higher loss
        # so we will return 1 - accuracy
        rem_idx = np.array(self.remaining_idx)
        if len(self.observed_idx) == 0:
            # No surrogate, so return uniform loss
            return np.ones(len(rem_idx), dtype=np.float64)
        model_probs = self.model_preds[rem_idx]
        surrogate_probs = self.surrogate_preds[rem_idx]

        pred_classes = np.argmax(model_probs, axis=1)

        res = 1 - surrogate_probs[np.arange(len(surrogate_probs)), pred_classes]

        return np.maximum(res, np.max(res) * 0.05)

    def sample(self):
        expected_loss = self.accuracy_loss()

        if (expected_loss < 0).sum() > 0:
            # Log-lik can be negative.
            # Make all values positive.
            # Alternatively could set <0 values to 0.
            expected_loss += np.abs(expected_loss.min())

        if expected_loss.sum() != 0:
            expected_loss /= expected_loss.sum()

        # Sample according to the expected loss pmf
        sample = self.sample_pmf(expected_loss)
        self.update_surrogate()
        return sample


class InformationGainSampler(AcquisitionFunction):
    """
    Acquisition function that selects samples based on information gain,
    defined as the cross-entropy between the surrogate model's and main model's
    predicted class probabilities.

    At each step, computes the cross-entropy for each unlabelled sample and
    samples according to a normalized probability over these values.

    Args:
        data (dict[str, Dataset]): Dictionary containing datasets.
        eval_dir (str): Directory for evaluation outputs.
        sampling_seed (int): Random seed for reproducibility.

    Methods:
        set_model_preds(preds): Sets the model's predicted probabilities.
        surrogate_pretrain(): Pre-trains the surrogate model if not already trained.
        update_surrogate(): Updates surrogate predictions based on labelled data.
        information_gain(): Computes information gain for unlabelled samples.
        sample(): Selects the next sample to label based on information gain.
    """

    def __init__(self, data: Dataset, eval_dir: str, sampling_seed: int):
        """
        Samples based on information gain (cross-entropy between surrogate and model
        predictions).

        Args:
            data: The dataset to sample from.
        """
        self.n_sampled = 0
        self.rng = np.random.default_rng(seed=sampling_seed)
        self.eval_data = data["evaluation_dataset"]
        self.surrogate_train_data = data["surrogate_train_dataset"]
        self.observed_idx: list[int] = []
        self.remaining_idx: list[int] = np.arange(len(self.eval_data)).tolist()
        self.eval_dir = eval_dir

        # Get embeddings for use with surrogate classifier
        self.embeddings = get_distilbert_embeddings(self.eval_data, eval_dir)

        # initialise surrogate predictions and model
        self.num_classes = len(np.unique(self.eval_data["label"]))
        self.surrogate_preds = np.full(
            (len(self.eval_data), self.num_classes), 1.0 / self.num_classes
        )
        self.surrogate_model = SGDClassifier(
            loss="log_loss", random_state=self.rng.integers(1e9)
        )

        # Placeholder for model predictions (should be set externally)
        self.model_preds = np.zeros_like(self.surrogate_preds)
        self.name = "random_forest_ig"

    def set_model_preds(self, preds: np.ndarray):
        """Set model predictions externally (shape: [n_samples, n_classes])."""
        self.model_preds = softmax(preds)

    def surrogate_pretrain(self):
        X_train = get_distilbert_embeddings(
            self.surrogate_train_data,
            self.eval_dir,
            embedding_savename="surrogate_embeddings",
        )
        y_train = np.array(self.surrogate_train_data["label"])
        surrogate_path = os.path.join(
            self.eval_dir,
            f"pretrained_surrogate_{self.surrogate_model.__class__.__name__}.joblib",
        )
        if os.path.exists(surrogate_path):
            print(f"Loading pre-trained surrogate model from {surrogate_path} ...")
            self.surrogate_model = joblib.load(surrogate_path)
        else:
            print(f"Pre-training surrogate model and saving to {surrogate_path} ...")
            self.surrogate_model = self.surrogate_model.fit(X_train, y_train)
            joblib.dump(self.surrogate_model, surrogate_path)

    def update_surrogate(self):
        if len(self.observed_idx) == 0:
            self.surrogate_preds = np.zeros_like(self.surrogate_preds)
            return
        X_acquired = self.embeddings[np.array(self.observed_idx)]
        y_acquired = np.array(self.eval_data["label"])[np.array(self.observed_idx)]

        # # Concatenate surrogate_train_data points before model fitting
        # X_training = get_distilbert_embeddings(
        #     self.surrogate_train_data,
        #     self.eval_dir,
        #     embedding_savename="surrogate_embeddings",
        # )
        # y_training = np.array(self.surrogate_train_data["label"])
        # X_data = np.concatenate([X_acquired, X_training], axis=0)
        # y_data = np.concatenate([y_acquired, y_training], axis=0)

        self.surrogate_model = self.surrogate_model.partial_fit(X_acquired, y_acquired)
        # Get surrogate predictions for all samples
        intermediate_surrogate_preds = self.surrogate_model.predict_proba(
            self.embeddings
        )
        classes_ = self.surrogate_model.classes_
        # Fill a full array with zeros for all classes
        full_proba = np.zeros((len(self.eval_data), self.num_classes))
        # fill the full array with surrogate predictions
        # for each class in the surrogate predictions
        for i, c in enumerate(classes_):
            full_proba[:, c] = intermediate_surrogate_preds[:, i]
        self.surrogate_preds = full_proba

    def information_gain(self):
        rem_idx = np.array(self.remaining_idx)
        surrogate_probs = self.surrogate_preds[rem_idx]  # π(y | x)
        model_probs = self.model_preds[rem_idx]  # f(x)_y

        # Add small epsilon for numerical stability in log
        eps = 1e-12
        log_model_probs = np.log(model_probs + eps)
        # Cross-entropy: -sum_y π(y|x) * log f(x)_y
        info_gain = -np.sum(surrogate_probs * log_model_probs, axis=1)
        # Make all values positive and nonzero for sampling
        return np.maximum(info_gain, np.max(info_gain) * 0.05)

    def sample(self):
        info_gain = self.information_gain()
        if (info_gain < 0).sum() > 0:
            info_gain += np.abs(info_gain.min())
        if info_gain.sum() != 0:
            info_gain /= info_gain.sum()
        # Sample according to the expected loss pmf
        sample = self.sample_pmf(info_gain)
        self.update_surrogate()
        return sample


class IsolationForestSampler(AcquisitionFunction):
    """
    Acquisition function that selects samples based on anomaly scores using an isolation
    forest in embedding space.

    Args:
        data (dict[str, Dataset]): Dictionary containing datasets.
        eval_dir (str): Directory for evaluation outputs.
        sampling_seed (int): Random seed for reproducibility.

    Methods:
        sample(): Selects the next sample to label based on anomaly score.
    """

    def __init__(self, data: dict[str, Dataset], eval_dir: str, sampling_seed: int):
        self.observed_idx: list[int] = []
        self.remaining_idx: list[int] = np.arange(
            len(data["evaluation_dataset"])
        ).tolist()
        self.n_sampled = 0
        self.rng = np.random.default_rng(seed=sampling_seed)
        self.eval_data = data["evaluation_dataset"]
        self.embeddings = get_distilbert_embeddings(self.eval_data, eval_dir)
        self.eval_dir = eval_dir
        self.name = "isolation_forest_sampler"

    def sample(self):
        if len(self.observed_idx) == 0:
            N = len(self.eval_data)
            pmf = np.ones(N, dtype=np.float64) / N
        else:
            observed = self.embeddings[np.array(self.observed_idx)]
            remaining = self.embeddings[np.array(self.remaining_idx)]
            iso = IsolationForest(random_state=self.rng.integers(1e9))
            iso.fit(observed)
            # Higher score = more normal, so invert for anomaly
            anomaly_scores = -iso.score_samples(remaining)
            pmf = softmax(anomaly_scores)
        return self.sample_pmf(pmf)


class MinorityClassSampler(AcquisitionFunction):
    """
    Acquisition function that selects samples based on the predicted probability
    of a specified minority class.

    Args:
        data (dict[str, Dataset]): Dictionary containing datasets.
        minority_class (int): The class label considered as the minority class.
        sampling_seed (int): Random seed for reproducibility.

    Methods:
        set_model_preds(preds): Sets the model's predicted probabilities.
        sample(): Selects the next sample to label based on minority class probability.
    """

    def __init__(
        self,
        data: dict[str, Dataset],
        minority_class: int,
        sampling_seed: int,
        **kwargs,
    ):
        self.eval_data = data["evaluation_dataset"]
        self.observed_idx: list[int] = []
        self.remaining_idx: list[int] = np.arange(len(self.eval_data)).tolist()
        self.n_sampled = 0
        self.rng = np.random.default_rng(seed=sampling_seed)
        self.minority_class = minority_class
        self.name = "minority_class_sampler"

    def set_model_preds(self, preds: np.ndarray):
        """
        Set model predictions externally (shape: [n_samples, n_classes]).
        """
        self.model_preds = softmax(preds)

    def sample(self):
        if self.model_preds is None:
            err_msg = "Model predictions must be set before sampling."
            raise ValueError(err_msg)
        rem_idx = np.array(self.remaining_idx)
        # Use the probability of the minority class for each unlabelled sample
        minority_probs = self.model_preds[rem_idx, self.minority_class]
        # Avoid all-zero pmf
        if np.all(minority_probs == 0):
            pmf = np.ones_like(minority_probs, dtype=np.float64)
        else:
            pmf = minority_probs
        pmf = np.asarray(pmf, dtype=np.float64)
        if pmf.sum() != 0:
            pmf /= pmf.sum()
        return self.sample_pmf(pmf)
