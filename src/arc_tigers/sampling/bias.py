import numpy as np


class LUREBiasCorrector:
    """
    Implements a bias correction weighting factor for active testing sampling.

    The BiasCorrector computes a weighting factor (v_m) for a selected sample
    during active testing, as described in the formula:

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
        N: Total number of samples in the dataset.
        M: Total number of samples to be labelled.

    Attributes:
        v_values: List to store the computed weighting factors for each sample.
    """

    def __init__(self, N: int, M: int):
        self.N = N
        self.M = M
        self.sample_weights: list[float] = []

    def compute_weighting_factor(self, q_im: float, m: int) -> float:
        """
        Compute the weighting factor v_m for the selected sample and update v_values.

        Args:
            q_im: The pmf value for the selected sample
            m: Number of samples labelled so far

        Returns:
            v_m: Weighting factor
        """
        inner = (1 / ((self.N - m + 1) * q_im)) - 1
        v_m = 1 + ((self.N - self.M) / (self.N - m)) * inner
        self.sample_weights.append(v_m)
        return v_m

    def apply_weighting_to_dict(
        self, q: float, m: int, metrics: dict[str, float]
    ) -> dict[str, float]:
        """
        Apply the weighting factor to all values in the metrics dictionary.

        Args:
            q: The pmf value for the selected sample
            m: Number of samples labelled so far
            metrics: Dictionary of metric values to be weighted {metric_name: value}

        Returns:
            Dictionary with weighted metric values.
        """

        weighting = self.compute_weighting_factor(q, m)
        return {k: v * weighting for k, v in metrics.items()}


class HTBiasCorrector:
    def __init__(self, sample_probabilities: list[float]):
        self.sample_weights = (1 / np.array(sample_probabilities)).tolist()
