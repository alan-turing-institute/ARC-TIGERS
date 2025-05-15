import numpy as np
from datasets import Dataset


class RandomSampler:
    def __init__(self, data: Dataset, seed: int | None = None):
        """Randomly samples a dataset.
        Args:
            data: The dataset to sample from.
            seed: The random seed for sampling. If None, a random seed will be used.

        Properties:
            n_sampled: Number of samples that have been labelled so far.
            labelled_idx: Indices of the samples that have been labelled.
            unlabelled_idx: Indices of the samples that haven't been labelled.
        """
        self.seed = seed
        # Random sampling has no dependencies on a model/similar, so we can pre-compute
        # the random order the dataset will be sampled in.
        rng = np.random.default_rng(seed)
        self._sample_order = rng.permutation(np.arange(len(data)))
        self.n_sampled = 0

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

    def sample(self) -> int:
        """
        Updates the n_sampled counter and gets the next sample to label.

        Returns:
            Index of the next sample to be labelled
        """
        self.n_sampled += 1
        if self.n_sampled > len(self._sample_order):
            msg = "No more samples to sample."
            raise ValueError(msg)
        return self._sample_order[self.n_sampled - 1]
