import numpy as np
from datasets import Dataset


class RandomSampler:
    def __init__(self, data: Dataset, seed: int | None = None):
        self.seed = seed
        self._rng = np.random.default_rng(seed)
        self._sample_order = self._rng.permutation(np.arange(len(data)))
        self._n_sampled = 0

    @property
    def labelled_idx(self) -> list[int]:
        return self._sample_order[: self._n_sampled]

    @property
    def unlabelled_idx(self) -> list[int]:
        return self._sample_order[self._n_sampled :]

    def sample(self) -> int:
        self._n_sampled += 1
        return self._sample_order[self._n_sampled - 1]
