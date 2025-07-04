from abc import ABC, abstractmethod

import numpy as np
from datasets import Dataset


class Sampler(ABC):
    """
    Abstract base class for samplers.

    Args:
        eval_data: The dataset to sample from.
        seed: The random seed for sampling. If None, a random seed will be used.

    Properties:
        labelled_idx: Indices of the samples that have been labelled.
        unlabelled_idx: Indices of the samples that haven't been labelled.
    """

    def __init__(self, eval_data: Dataset, seed: int, name: str):
        """ """
        self.eval_data = eval_data
        self.rng = np.random.default_rng(seed)
        self.name = name

    @property
    @abstractmethod
    def labelled_idx(self) -> list[int]:
        """
        Returns:
            Indices of the samples that have been labelled.
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def unlabelled_idx(self) -> list[int]:
        """
        Returns:
            Indices of the samples that haven't been labelled.
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def sample_prob(self) -> list[float]:
        """
        Returns:
            Probabilities of the samples being selected for labelling.
        """
        raise NotImplementedError

    @abstractmethod
    def sample(self) -> tuple[int, float]:
        """
        Updates the n_sampled counter and gets the next sample to label.

        Returns:
            Index of the next sample to be labelled, and the probability of that sample
            being selected.
        """
        raise NotImplementedError
