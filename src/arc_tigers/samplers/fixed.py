from datasets import Dataset

from arc_tigers.samplers.sampler import Sampler


class FixedSampler(Sampler):
    """Sampler that uses a predefined sequence of sample indices and their
    corresponding probabilities.

    This sampler follows a fixed, predetermined sampling order. It allows you to
    specify exactly which samples to select and in what order, along with their
    associated probabilities. This is useful for replicating specific sampling
    strategies or for testing with predetermined sampling sequences.

    Args:
        eval_data: The dataset to sample from.
        seed: The random seed for sampling. If None, a random seed will be used.
        sample_indices: List of sample indices in the order they should be
            sampled.
        sample_probabilities: List of probabilities corresponding to each sample
            index. Must have the same length as sample_indices.

    Properties:
        n_sampled: Number of samples that have been labelled so far.
        labelled_idx: Indices of the samples that have been labelled.
        unlabelled_idx: Indices of the samples that haven't been labelled.
    """

    def __init__(
        self,
        eval_data: Dataset,
        seed: int,
        sample_indices: list[int],
        sample_probabilities: list[float],
        **kwargs,
    ):
        super().__init__(eval_data, seed, "fixed")

        # Validate inputs
        if len(sample_indices) != len(sample_probabilities):
            msg = (
                f"Length of sample_indices ({len(sample_indices)}) must match "
                f"length of sample_probabilities ({len(sample_probabilities)})"
            )
            raise ValueError(msg)

        if not all(0 <= idx < len(eval_data) for idx in sample_indices):
            msg = f"All sample indices must be valid (0 <= idx < {len(eval_data)})"
            raise ValueError(msg)

        if not all(0 <= prob <= 1 for prob in sample_probabilities):
            msg = "All sample probabilities must be between 0 and 1"
            raise ValueError(msg)

        self.n_sampled = 0
        self._sample_order = sample_indices.copy()
        self._sample_prob = sample_probabilities.copy()

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
