from collections import namedtuple
import numpy as np
from datasets import Dataset
from scipy.spatial.distance import cdist
from sklearn.ensemble import RandomForestClassifier
from arc_tigers.eval.utils import softmax
from arc_tigers.sample.utils import get_distilbert_embeddings

DummyAcqFunc = namedtuple("DummyAcqFunc", ["observed_idx", "remaining_idx", "rng"])


class BiasCorrector:
    """
    Implements a bias correction weighting factor for active learning sampling.

    The BiasCorrector computes a weighting factor (v_m) for a selected sample
    during active learning, as described in the formula:

        v_m = 1 + (N - M) / (N - m) * (1 / ((N - m + 1) * q_im) - 1)

    where:
        N    : Total number of samples in the dataset.
        M    : Total number of samples to be labelled (target number).
        m    : Number of samples labelled so far.
        q_im : The probability (from the acquisition function's pmf) of selecting the
        sample at step m.

    This correction is used to adjust for the sampling bias introduced by non-uniform
    acquisition functions, ensuring unbiased estimation of metrics or statistics
    over the dataset.

    Args:
        N (int): Total number of samples in the dataset.
        M (int): Total number of samples to be labelled.

    Methods:
        call(q_im, m): Computes the weighting factor for the selected sample.
    """

    def __init__(self, N, M):
        self.N = N
        self.M = M

    def compute_weighting_factor(self, q_im, m):
        """
        Compute the weighting factor v_m for the selected sample.

        Args:
            q_im: The pmf value for the selected sample (float)
            N: Total number of samples (int)
            M: Total number of samples to be labelled (int)
            m: Number of samples labelled so far (int)

        Returns:
            v_m: Weighting factor (float)
        """
        inner = (1 / ((self.N - m + 1) * q_im)) - 1
        return 1 + ((self.N - self.M) / (self.N - m)) * inner

    def apply_weighting_to_dict(
        self, q: float, m: int, metrics: dict[str, float]
    ) -> dict[str, float]:
        """
        Apply the weighting factor to all values in the metrics dictionary.

        Args:
            q: The pmf value for the selected sample (float)
            m: Number of samples labelled so far (int)
            metrics: Dictionary of metric values to be weighted {metric_name: value}

        Returns:
            dict: Dictionary with weighted metric values.
        """

        weighting = self.compute_weighting_factor(q, m)
        return {k: v * weighting for k, v in metrics.items()}


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

    def __init__(self, data: Dataset, sampling_seed: int, eval_dir: str):
        """Samples a dataset based on distance between observed and unobserved
        embeddings.

        Args:
            data: The dataset to sample from.

        Properties:
            n_sampled: Number of samples that have been labelled so far.
            labelled_idx: Indices of the samples that have been labelled.
            unlabelled_idx: Indices of the samples that haven't been labelled.
        """
        self.observed_idx: list[int] = []
        self.remaining_idx: list[int] = np.arange(len(data)).tolist()
        self.n_sampled = 0
        self.rng = np.random.default_rng(seed=sampling_seed)

        self.data = data
        # Get embeddings for whole dataset
        self.embeddings = get_distilbert_embeddings(data, eval_dir)

    def sample(self) -> int:
        """
        Updates the n_sampled counter and gets the next sample to label.

        Returns:
            Index of the next sample to be labelled
        """

        # If no samples have been selected yet, sample from a uniform distribution
        if len(self.observed_idx) == 0:
            N = len(self.data)
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


class RFSampler(AcquisitionFunction):
    """
    Acquisition function that selects samples based on disagreement between
    a surrogate Random Forest classifier and the main model.

    At each step, computes the expected loss (1 - surrogate accuracy) for each
    unlabelled sample, and samples according to a normalized probability over these
    losses.

    Args:
        data (Dataset): The dataset to sample from.
        eval_dir (str): Directory for evaluation outputs.
        sampling_seed (int): Random seed for reproducibility.

    Methods:
        set_model_preds(preds): Sets the model's predicted probabilities.
        accuracy_loss(): Computes expected loss for unlabelled samples.
        sample(): Selects the next sample to label based on expected loss.
    """

    def __init__(self, data: Dataset, eval_dir: str, sampling_seed: int):
        """Samples a dataset based on distance between model loss and surrogate loss,
        where the surrogate is a Random Forest classifier.

        Args:
            data: The dataset to sample from.

        Properties:
            n_sampled: Number of samples that have been labelled so far.
            labelled_idx: Indices of the samples that have been labelled.
            unlabelled_idx: Indices of the samples that haven't been labelled.
        """
        self.observed_idx: list[int] = []
        self.remaining_idx = np.arange(len(data)).tolist()
        self.n_sampled = 0
        self.model_preds = np.array([])
        self.surrogate_preds = np.array([])
        self.data = data
        # Get embeddings for use with RF classifier
        self.embeddings = get_distilbert_embeddings(data, eval_dir)
        self.rng = np.random.default_rng(seed=sampling_seed)

        # initialise surrogate predictions
        self.num_classes = len(np.unique(self.data["label"]))
        self.surrogate_preds = np.full(
            (len(self.data), self.num_classes), 1.0 / self.num_classes
        )

        # Placeholder for model predictions (should be set externally)
        self.model_preds = np.zeros_like(self.surrogate_preds)

    def set_model_preds(self, preds: np.ndarray):
        """Set model predictions externally (shape: [n_samples, n_classes])."""
        self.model_preds = softmax(preds)

    def update_surrogate(self):
        if len(self.observed_idx) == 0:
            self.surrogate_preds = np.zeros_like(self.surrogate_preds)
            return
        X_train = self.embeddings[np.array(self.observed_idx)]
        y_train = np.array(self.data["label"])[np.array(self.observed_idx)]
        self.rf_classifier = RandomForestClassifier(
            random_state=self.rng.integers(1e9), n_estimators=100, n_jobs=-1
        ).fit(X_train, y_train)
        # Get surrogate predictions for all samples
        intermediate_surrogate_preds = self.rf_classifier.predict_proba(self.embeddings)
        classes_ = self.rf_classifier.classes_
        # Fill a full array with zeros for all classes
        full_proba = np.zeros((len(self.data), self.num_classes))
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
    measured as the cross-entropy between the surrogate model's and main model's
    predicted class probabilities.

    At each step, computes the cross-entropy for each unlabelled sample and
    samples according to a normalized probability over these values.

    Args:
        data (Dataset): The dataset to sample from.
        eval_dir (str): Directory for evaluation outputs.
        sampling_seed (int): Random seed for reproducibility.

    Methods:
        set_model_preds(preds): Sets the model's predicted probabilities.
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
        self.observed_idx: list[int] = []
        self.remaining_idx: list[int] = np.arange(len(data)).tolist()
        self.n_sampled = 0
        self.rng = np.random.default_rng(seed=sampling_seed)
        self.data = data

        # Get embeddings for use with surrogate classifier
        self.embeddings = get_distilbert_embeddings(data, eval_dir)

        # initialise surrogate predictions
        self.num_classes = len(np.unique(self.data["label"]))
        self.surrogate_preds = np.full(
            (len(self.data), self.num_classes), 1.0 / self.num_classes
        )

        # Placeholder for model predictions (should be set externally)
        self.model_preds = np.zeros_like(self.surrogate_preds)

    def set_model_preds(self, preds: np.ndarray):
        """Set model predictions externally (shape: [n_samples, n_classes])."""
        self.model_preds = softmax(preds)

    def update_surrogate(self):
        if len(self.observed_idx) == 0:
            self.surrogate_preds = np.zeros_like(self.surrogate_preds)
            return
        X_train = self.embeddings[np.array(self.observed_idx)]
        y_train = np.array(self.data["label"])[np.array(self.observed_idx)]
        self.rf_classifier = RandomForestClassifier(
            random_state=self.rng.integers(1e9), n_estimators=100, n_jobs=-1
        ).fit(X_train, y_train)
        # Get surrogate predictions for all samples
        intermediate_surrogate_preds = self.rf_classifier.predict_proba(self.embeddings)
        classes_ = self.rf_classifier.classes_
        # Fill a full array with zeros for all classes
        full_proba = np.zeros((len(self.data), self.num_classes))
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
