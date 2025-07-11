import numpy as np
from datasets import Dataset

from arc_tigers.samplers.sampler import Sampler


class RandomSampler(Sampler):
    """Randomly samples a dataset.
    Args:
        eval_data: The dataset to sample from.
        seed: The random seed for sampling. If None, a random seed will be used.

    Properties:
        n_sampled: Number of samples that have been labelled so far.
        labelled_idx: Indices of the samples that have been labelled.
        unlabelled_idx: Indices of the samples that haven't been labelled.
    """

    def __init__(self, eval_data: Dataset, seed: int, **kwargs):
        super().__init__(eval_data, seed, "random")
        # Random sampling has no dependencies on a model/similar, so we can pre-compute
        # the random order the dataset will be sampled in.
        self.n_sampled = 0
        self._sample_order = self.rng.permutation(np.arange(len(eval_data))).tolist()
        self._sample_prob = (np.arange(1, len(eval_data) + 1) / len(eval_data)).tolist()

    @property
    def labelled_idx(self) -> list[int]:
        """
        Returns:
            Indices of the samples that have been labelled.
        """
        return self._sample_order[: self.n_sampled]

    @property
    def unlabelled_idx(self) -> list[int]:
        """
        Returns:
            Indices of the samples that haven't been labelled.
        """
        return self._sample_order[self.n_sampled :]

    @property
    def sample_prob(self) -> list[float]:
        """
        Returns:
            Probabilities of the samples being selected for labelling.
        """
        return self._sample_prob[: self.n_sampled]

    def sample(self) -> tuple[int, float]:
        """
        Updates the n_sampled counter and gets the next sample to label.

        Returns:
            Index of the next sample to be labelled, and the probability of that sample
            being selected.
        """
        self.n_sampled += 1
        if self.n_sampled > len(self._sample_order):
            msg = "No more samples to sample."
            raise IndexError(msg)
        return self._sample_order[self.n_sampled - 1], self._sample_prob[
            self.n_sampled - 1
        ]
