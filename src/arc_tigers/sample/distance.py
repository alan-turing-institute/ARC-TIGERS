import numpy as np
from datasets import Dataset

from arc_tigers.sample.utils import get_embeddings


class DistanceSampler:
    def __init__(self, data: Dataset):
        """Samples a dataset based on distance between observed and unobserved
        embeddings.

        Args:
            data: The dataset to sample from.

        Properties:
            n_sampled: Number of samples that have been labelled so far.
            labelled_idx: Indices of the samples that have been labelled.
            unlabelled_idx: Indices of the samples that haven't been labelled.
        """
        self.observed_idx = []
        self.remaining_idx = np.arange(len(data))
        self.n_sampled = 0

        # Get embeddings for whole dataset
        self.data = data
        self.embeddings = get_embeddings(data)

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

    def sample(self) -> int:
        """
        Updates the n_sampled counter and gets the next sample to label.

        Returns:
            Index of the next sample to be labelled
        """

        # If no samples have been selected yet, sample from a uniform distribution
        if len(self.observed_idx) == 0:
            N = len(self.data)
            pmf = np.ones(N) / N

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

        # plt.plot(pmf)
        sample = np.random.multinomial(1, pmf)
        idx = np.where(sample)[0][0]
        test_idx = self.remaining_idx[idx]
        self.observed_idx.append(int(test_idx))

        return test_idx
