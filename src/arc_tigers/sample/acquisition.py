import logging
import os
from abc import abstractmethod

import joblib
import numpy as np
import numpy.typing as npt
from datasets import Dataset
from scipy.spatial.distance import cdist
from sklearn.ensemble import IsolationForest, RandomForestClassifier

from arc_tigers.eval.utils import softmax
from arc_tigers.sample.sampler import Sampler
from arc_tigers.sample.utils import SurrogateData

logger = logging.getLogger(__name__)


class AcquisitionFunction(Sampler):
    """
    Abstract base class for active testing-like strategies that sample from a pmf
    created from an acquisition function.

    See Sampler class for documentation of common functionality.
    """

    def __init__(self, eval_data: Dataset, seed: int, name: str):
        super().__init__(eval_data, seed, name)
        self.remaining_idx: list[int] = np.arange(len(self.eval_data)).tolist()
        self.observed_idx: list[int] = []
        self._sample_prob: list[float] = []

    @property
    def sample_prob(self) -> list[float]:
        return self._sample_prob

    @property
    def labelled_idx(self) -> list[int]:
        return self.observed_idx

    @property
    def unlabelled_idx(self) -> list[int]:
        return self.remaining_idx

    def sample_pmf(self, pmf: npt.NDArray[np.floating]) -> tuple[int, float]:
        """
        Samples an index from the unlabelled pool according to a pmf.

        Args:
            pmf: Probability mass function (pmf) to sample from.

        Returns:
            A tuple containing the index of the sampled item and the probability of that
            item being selected.
        """
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
        idx: int = np.where(prob)[0][0]
        sample_prob = pmf[idx]
        test_idx = self.remaining_idx[idx]
        self.observed_idx.append(int(test_idx))
        self._sample_prob.append(sample_prob)
        self.remaining_idx.remove(int(test_idx))

        return test_idx, sample_prob


class AcquisitionFunctionWithEmbeds(AcquisitionFunction):
    """
    Abstract base class for acquisition strategies that use embeddings of the dataset
    to compute acquisition functions.

    See AcquisitionFunction for documentation of common functionality.

    Args:
        embed_dir: Directory where embeddings are stored or will be saved.

    Attributes:
        embed_dir: Directory where embeddings are stored or will be saved.
        eval_embed: Embeddings of the evaluation dataset.
    """

    def __init__(
        self, eval_data: Dataset, seed: int, name: str, eval_embeds: np.ndarray
    ):
        super().__init__(eval_data, seed, name)
        self.eval_embeds = eval_embeds


class DistanceSampler(AcquisitionFunctionWithEmbeds):
    """
    Sampler that selects samples based on their average distance (in embedding space)
    to the currently labelled set. At each step, computes the mean Euclidean distance
    from each unlabelled sample to all labelled samples, and samples according to a
    softmax over these distances.

    See AcquisitionFunctionWithEmbeds for documentation of common functionality.
    """

    def __init__(
        self, eval_data: Dataset, seed: int, eval_embeds: np.ndarray, **kwargs
    ):
        super().__init__(eval_data, seed, "distance", eval_embeds)

    def sample(self) -> tuple[int, float]:
        """Sample based on embedding distance to previously labelled samples."""

        # If no samples have been selected yet, sample from a uniform distribution
        if len(self.observed_idx) == 0:
            N = len(self.eval_data)
            # float16 causes an underflow error for large datasets
            pmf = np.ones(N, dtype=np.float64) / N

        else:
            # Subset embeddings
            remaining = self.eval_embeds[np.array(self.remaining_idx)]
            observed = self.eval_embeds[np.array(self.observed_idx)]

            # Compute all pairwise Euclidean distances
            dists = cdist(remaining, observed, metric="euclidean")
            # Mean distance to observed for each remaining sample
            distances = dists.mean(axis=1)

            # Constract PDF via softmax
            pmf = np.exp(distances)
            pmf /= pmf.sum()

        return self.sample_pmf(pmf)


class SurrogateSampler(AcquisitionFunctionWithEmbeds):
    """
    Abstract base class for acquisition strategies that select samples based on
    disagreement between a surrogate classifier (random forest) and the main model.

    See AcquisitionFunctionWithEmbeds for documentation of common functionality.

    Args:
        model_preds: Predicted scores from the main model.
        surrogate_data: Dataset used to pre-train the surrogate model (or None).

    Attributes:
        surrogate_model: Trained surrogate classifier (Random Forest).
        surrogate_preds: Predicted probabilities from the surrogate model.
        num_classes: Number of classes in the dataset.
        surrogate_train_data: Dataset used to pre-train the surrogate (if provided).
        train_embeds: Embeddings of the surrogate training data (if provided).
    """

    def __init__(
        self,
        eval_data: Dataset,
        seed: int,
        name: str,
        eval_embeds: np.ndarray,
        model_preds: np.ndarray,
        surrogate_train_data: SurrogateData | None = None,
        **kwargs,
    ):
        super().__init__(eval_data, seed, name, eval_embeds)

        if len(model_preds) != len(eval_data):
            err_msg = (
                f"Model predictions length {len(model_preds)} does not match "
                f"evaluation data length {len(eval_data)}."
            )
            raise ValueError(err_msg)
        self.model_preds = softmax(model_preds)  # [n_samples, n_classes]

        # initialise surrogate predictions and model
        self.num_classes = len(np.unique(self.eval_data["label"]))
        self.surrogate_preds = np.full(
            (len(self.eval_data), self.num_classes), 1.0 / self.num_classes
        )
        self.surrogate_model = RandomForestClassifier(
            random_state=self.rng.integers(int(1e9))
        )
        self.surrogate_train_data = surrogate_train_data
        self.surrogate_pretrain()

    def surrogate_pretrain(self):
        """Pre-trains the surrogate model if training data has been provided."""
        if self.surrogate_train_data is None:
            logger.warning(
                "No surrogate training data provided. Surrogate model will not be "
                "pre-trained, and no training data will be used for surrogate updates."
            )
            return

        surrogate_path = (
            self.surrogate_train_data.cache_dir
            / "surrogate"
            / f"{self.surrogate_model.__class__.__name__}.joblib"
        )
        if os.path.exists(surrogate_path):
            logger.info("Loading pre-trained surrogate model from %s", surrogate_path)
            self.surrogate_model = joblib.load(surrogate_path)
        else:
            logger.info(
                "Pre-training surrogate model and saving to %s...", surrogate_path
            )
            self.surrogate_model = self.surrogate_model.fit(
                self.surrogate_train_data.embed, self.surrogate_train_data.label
            )
            os.makedirs(surrogate_path.parent, exist_ok=True)
            joblib.dump(self.surrogate_model, surrogate_path)

    def update_surrogate(self):
        """Updates the surrogate model with newly labelled data, and re-computes its
        predictions on the unlabelled data."""
        if len(self.observed_idx) == 0:
            logger.warning(
                "No samples have been labelled. Surrogate model will not be updated."
            )
            return

        train_X = self.eval_embeds[np.array(self.observed_idx)]
        train_y = np.array(self.eval_data["label"])[np.array(self.observed_idx)]

        if self.surrogate_train_data is not None:
            train_X = np.concat((self.surrogate_train_data.embed, train_X))
            train_y = np.concat((self.surrogate_train_data.label, train_y))

        self.surrogate_model = self.surrogate_model.fit(train_X, train_y)

        # Get surrogate predictions for all samples
        intermediate_surrogate_preds = self.surrogate_model.predict_proba(
            self.eval_embeds
        )
        classes_ = self.surrogate_model.classes_
        # Fill a full array with zeros for all classes
        full_proba = np.zeros((len(self.eval_data), self.num_classes))
        # fill the full array with surrogate predictions
        # for each class in the surrogate predictions
        for i, c in enumerate(classes_):
            full_proba[:, c] = intermediate_surrogate_preds[:, i]
        self.surrogate_preds = full_proba

    @abstractmethod
    def acquisition_fn(
        self, model_probs: np.ndarray, surrogate_probs: np.ndarray
    ) -> np.ndarray:
        """
        Computes the acquisition function based on model and surrogate probabilities.
        This method must be implemented by subclasses.

        Args:
            model_probs: Predicted probabilities from the main model.
            surrogate_probs: Predicted probabilities from the surrogate model.

        Returns:
            Acquisition values for each sample.
        """
        raise NotImplementedError

    def sample(self) -> tuple[int, float]:
        """Sample according to the acquisition function values."""
        # sample from remaining (unlabelled) samples
        rem_idx = np.array(self.remaining_idx)
        model_probs = self.model_preds[rem_idx]
        surrogate_probs = self.surrogate_preds[rem_idx]

        q = self.acquisition_fn(model_probs, surrogate_probs)

        if (q < 0).sum() > 0:
            # Make all values positive.
            q += np.abs(q.min())

        if q.sum() != 0:
            q /= q.sum()

        # Sample according to the acquisition values
        sample = self.sample_pmf(q)
        self.update_surrogate()
        return sample


class AccSampler(SurrogateSampler):
    """
    Sampler that uses the acquisition function for accuracy from the active testing
    paper (equation 14).

    See SurrogateSampler for documentation of common functionality.
    """

    def __init__(
        self,
        eval_data: Dataset,
        seed: int,
        eval_embeds: np.ndarray,
        model_preds: np.ndarray,
        surrogate_train_data: SurrogateData | None = None,
        **kwargs,
    ):
        super().__init__(
            eval_data, seed, "accuracy", eval_embeds, model_preds, surrogate_train_data
        )

    def acquisition_fn(
        self, model_probs: np.ndarray, surrogate_probs: np.ndarray
    ) -> np.ndarray:
        """Accuracy acquisition function."""
        pred_classes = np.argmax(model_probs, axis=1)

        # we need higher values = higher loss so we will return 1 - accuracy
        res = 1 - surrogate_probs[np.arange(len(surrogate_probs)), pred_classes]

        return np.maximum(res, np.max(res) * 0.05)


class InformationGainSampler(SurrogateSampler):
    """
    Acquisition function that selects samples based on information gain,
    defined as the cross-entropy between the surrogate model's and main model's
    predicted class probabilities (equation 11 in the active testing paper).

    See SurrogateSampler for documentation of common functionality.
    """

    def __init__(
        self,
        eval_data: Dataset,
        seed: int,
        eval_embeds: np.ndarray,
        model_preds: np.ndarray,
        surrogate_train_data: SurrogateData | None = None,
        **kwargs,
    ):
        super().__init__(
            eval_data, seed, "info_gain", eval_embeds, model_preds, surrogate_train_data
        )

    def acquisition_fn(
        self, model_probs: np.ndarray, surrogate_probs: np.ndarray
    ) -> np.ndarray:
        """Information gain acquisition function."""
        # Add small epsilon for numerical stability in log
        eps = 1e-12
        log_model_probs = np.log(model_probs + eps)
        # Cross-entropy: -sum_y π(y|x) * log f(x)_y
        info_gain = -np.sum(surrogate_probs * log_model_probs, axis=1)
        # Make all values positive and nonzero for sampling
        return np.maximum(info_gain, np.max(info_gain) * 0.05)


class IsolationForestSampler(AcquisitionFunctionWithEmbeds):
    """
    Sampler that selects samples based on anomaly scores using an isolation forest in
    embedding space.

    See AcquisitionFunctionWithEmbeds for documentation of common functionality.
    """

    def __init__(
        self, eval_data: Dataset, seed: int, eval_embeds: np.ndarray, **kwargs
    ):
        super().__init__(eval_data, seed, "isolation", eval_embeds)

    def sample(self) -> tuple[int, float]:
        """
        Samples the next item to label based on anomaly scores of the unlabelled data,
        as computed by an isolation forest trained on the labelled data.
        """
        if len(self.observed_idx) == 0:
            N = len(self.eval_data)
            pmf = np.ones(N, dtype=np.float64) / N
        else:
            observed = self.eval_embeds[np.array(self.observed_idx)]
            remaining = self.eval_embeds[np.array(self.remaining_idx)]
            iso = IsolationForest(random_state=self.rng.integers(int(1e9)))
            iso.fit(observed)
            # Higher score = more normal, so invert for anomaly
            anomaly_scores = -iso.score_samples(remaining)
            pmf = softmax(anomaly_scores)
        return self.sample_pmf(pmf)


class MinorityClassSampler(AcquisitionFunction):
    """
    Sampler that selects samples based on the predicted probability of them belonging to
    the minority class, aiming to capture as many minority samples as possible on
    imbalanced datasets.

    See AcquisitionFunction for documentation of common functionality.

    Args:
        minority_class: The class label considered as the minority class.
        model_preds: Predicted class probabilities from the model being evaluated.
    """

    def __init__(
        self,
        eval_data: Dataset,
        minority_class: int,
        seed: int,
        model_preds: np.ndarray,
        **kwargs,
    ):
        super().__init__(eval_data, seed, "minority")
        self.minority_class = minority_class
        self.model_preds = softmax(model_preds)

    def sample(self) -> tuple[int, float]:
        """Sample based on the predicted probability of the minority class."""
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
