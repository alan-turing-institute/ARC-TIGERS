from collections import namedtuple

import numpy as np
from datasets import Dataset
from sklearn.ensemble import RandomForestClassifier

from arc_tigers.sample.utils import get_distilbert_embeddings

DummyAcqFunc = namedtuple("DummyAcqFunc", ["observed_idx", "remaining_idx", "rng"])


class AcquisitionFunction:
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

        sample = self.rng.multinomial(1, pmf)
        idx = np.where(sample)[0][0]
        test_idx = self.remaining_idx[idx]
        self.observed_idx.append(int(test_idx))

        return test_idx

    def sample(self) -> int:
        raise NotImplementedError


class DistanceSampler(AcquisitionFunction):
    def __init__(self, data: Dataset, sampling_seed: int):
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
        self.remaining_idx = np.arange(len(data))
        self.n_sampled = 0
        self.rng = np.random.default_rng(seed=sampling_seed)

        self.data = data
        # Get embeddings for whole dataset
        self.embeddings = get_distilbert_embeddings(data)

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
            remaining = self.embeddings[self.remaining_idx]
            observed = self.embeddings[self.observed_idx]

            # Broadcasting to get all paired differences
            d = remaining[:, np.newaxis, :] - observed
            d = d**2
            # Sum over feature dimension
            d = d.sum(-1)
            # sqrt to get distance
            d = np.sqrt(d)
            # Mean over other pairs
            distances = d.mean(1)

            # Constract PDF via softmax
            pmf = np.exp(distances)
            pmf /= pmf.sum()

        return self.sample_pmf(pmf)


class RFSampler(AcquisitionFunction):
    def __init__(self, data: Dataset):
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
        self.remaining_idx = np.arange(len(data))
        self.n_sampled = 0
        self.model_preds = np.array([])
        self.surrogate_preds = np.array([])
        self.data = data
        # Get embeddings for use with RF classifier
        self.embeddings = get_distilbert_embeddings(data)

        # Train Random Forest classifier
        self.rf_classifier = RandomForestClassifier().fit(
            self.embeddings, self.data["label"]
        )

        # TODO: Get predictions for model and surrogate over all remaining *test* points
        # self.model_preds =
        # self.surrogate_preds =

    def accuracy_loss(self):
        # we need higher values = higher loss
        # so we will return 1 - accuracy

        pred_classes = np.argmax(self.model_preds, axis=1)

        # instead of 0,1 loss we get p_surr(y|x) for accuracy

        res = (
            1 - self.surrogate_preds[np.arange(len(self.surrogate_preds)), pred_classes]
        )

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

        return self.sample_pmf(expected_loss)
