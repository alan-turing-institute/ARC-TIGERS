"""
Implementation of synthetic models using beta distributions.
"""

import argparse
from collections.abc import Callable

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import seaborn as sns
import torch
from scipy.optimize import minimize
from scipy.stats import beta
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    precision_recall_fscore_support,
)
from torch import nn
from transformers.modeling_outputs import TokenClassifierOutput


def fit_positive_distribution(
    positive_err_rate: float,
    b: float = 2.0,
    force_a_gt_b: bool = False,
    tol: float = 1e-9,
) -> tuple[float, float]:
    """
    Fit a beta distribution to sample simulated model scores for the positive class.
    Fits a beta distribution with "b" parameter value fixed by the "b" argument, and the
    "a" parameter optimized so that the CDF at 0.5 equals the positive error rate.

    Args:
        positive_err_rate: Error rate for the positive class (proportion of positive
            samples that will be assigned a predicted score < 0.5)
        b: Value for the beta distribution parameter b
        force_a_gt_b: If True, force a > b in the optimization (may help to give a beta
            distribution with a more reasonable shape)
        tol: Tolerance for optimization

    Returns:
        a, b values for the optimized beta distribution
    """

    def loss_positive(a: float) -> float:
        if force_a_gt_b and a < b:
            return np.inf
        if a <= 0:
            return np.inf
        candidate = beta(a, b)
        return (candidate.cdf(0.5) - positive_err_rate) ** 2

    a_init = b + 0.1
    result = minimize(loss_positive, a_init, tol=tol)
    if not result.success:
        msg = f"Optimization failed:\n{result}"
        raise ValueError(msg)

    return float(result.x[0]), b


def fit_negative_distribution(
    negative_err_rate: float,
    a: float = 2.0,
    force_b_gt_a: bool = False,
    tol: float = 1e-9,
) -> tuple[float, float]:
    """
    Fit a beta distribution to sample simulated model scores for the negative class.
    Fits a beta distribution with "a" parameter value fixed by the "a" argument, and the
    "b" parameter optimized so that the CDF at 0.5 equals the one minus the negative
    error rate.

    Args:
        negative_err_rate: Error rate for the positive class (proportion of negative
            samples that will be assigned a predicted score > 0.5)
        a: Value for the beta distribution parameter a
        force_b_gt_a: If True, force b > a in the optimization (may help to give a beta
            distribution with a more reasonable shape)
        tol: Tolerance for optimization

    Returns:
        a, b values for the optimized beta distribution
    """

    def loss_negative(b: float) -> float:
        if force_b_gt_a and b < a:
            return np.inf
        if b <= 0:
            return np.inf
        candidate = beta(a, b)
        return (candidate.cdf(0.5) - (1 - negative_err_rate)) ** 2

    b_init = a + 0.1
    result = minimize(loss_negative, b_init, tol=tol)
    if not result.success:
        msg = f"Optimization failed:\n{result}"
        raise ValueError(msg)

    return a, float(result.x[0])


def get_model(
    a_positive: float, b_positive: float, a_negative: float, b_negative: float
) -> Callable[
    [npt.NDArray[np.integer] | torch.Tensor],
    npt.NDArray[np.floating] | torch.Tensor,
]:
    """
    Get a model that simulates predictions based on fitted beta distributions for each
    class.

    Args:
        a_positive: Parameter a for the beta distribution for the positive class
        b_positive: Parameter b for the beta distribution for the positive class
        a_negative: Parameter a for the beta distribution for the negative class
        b_negative: Parameter b for the beta distribution for the negative class

    Returns:
        A callable that takes a label (0 or 1) and returns a simulated model score based
        on the beta distribution parameters.
    """

    def model(labels: npt.NDArray[np.integer]) -> npt.NDArray[np.floating]:
        scores = np.zeros_like(labels, dtype=float)
        neg_labels = labels == 0
        scores[neg_labels] = np.random.beta(a_negative, b_negative, neg_labels.sum())
        pos_labels = labels == 1
        scores[pos_labels] = np.random.beta(a_positive, b_positive, pos_labels.sum())

        return scores

    return model


def imbalance_scaled_error_rate(
    model_adv: float, imbalance: float
) -> tuple[float, float]:
    """
    Get class error rates based on a rough scaling of difficulty with dataset imbalance.

    Scales error rates as follows:
    - Positive class: (1 - imbalance) / model_adv
    - Negative class: imbalance / model_adv

    WARNING: This is just a starting point and bakes in strong assumptions about how
    difficulty scales with imbalance. It may be better to set the positive and negative
    error rates manually instead.

    Args:
        model_adv: Parameterization of model performance. model_adv = 1 aims to roughly
            approximate a model that always predicts the majority class. model_adv > 1
            is better than that baseline. On a balanced dataset (imbalance = 0.5),
            model_adv = 50 yields 1% error rates for each class (50 times better than
            random).
        imbalance: Fraction of positive labels in the dataset.

    Returns:
        positive_err_rate: Error rate for the positive class
        negative_err_rate: Error rate for the negative class
    """
    positive_err_rate = (1 - imbalance) / model_adv
    negative_err_rate = imbalance / model_adv
    return positive_err_rate, negative_err_rate


def get_model_stats(
    imbalance: float,
    model: Callable[
        [npt.NDArray[np.integer] | torch.IntTensor],
        npt.NDArray[np.floating] | torch.FloatTensor,
    ],
    plot: bool = True,
    verbose: bool = True,
) -> dict[str, float | np.ndarray]:
    """
    Simulate a dataset and compute model performance metrics.

    Args:
        imbalance: Fraction of positive labels in the dataset.
        model: Model to simulate predictions.
        plot: If True, plot the distribution of scores for each class.
        verbose: If True, print the model performance metrics.

    Returns:
        Model performance metrics: "precision", "recall", "f1", "ap" (average
        precision), and "cm" (confusion matrix).
    """
    N = max(1000, round(100 / imbalance))
    labels = (np.random.random(N) < imbalance).astype(int)
    preds = model(labels)
    pred_labels = (np.array(preds) > 0.5).astype(int)

    precision, recall, f1, _ = precision_recall_fscore_support(labels, pred_labels)
    ap = average_precision_score(labels, preds)
    cm = confusion_matrix(labels, pred_labels, normalize="all")

    stats_str = (
        f"Class 0 ({1 - imbalance:.0%} of data): "
        f"Pre={precision[0]:.2f}, Rec={recall[0]:.2f}, F1={f1[0]:.2f}\n"
        f"Class 1 ({imbalance:.0%} of data): "
        f"Pre={precision[1]:.2f}, Rec={recall[1]:.2f}, F1={f1[1]:.2f}\n"
        f"AP: {ap:.2f}\n"
        f"TN: {cm[0, 0]:.3f}, FP: {cm[0, 1]:.3f}, "
        f"FN: {cm[1, 0]:.3f}, TP: {cm[1, 1]:.3f}"
    )
    if verbose:
        print(stats_str)

    if plot:
        sns.kdeplot(x=preds, hue=labels, common_norm=False, clip=(0, 1))
        plt.axvline(0.5, color="k", linestyle="--")
        plt.title(f"Per-Class Score Distribution\n{stats_str}")
        plt.xlim(0, 1)
        plt.xlabel("Score")
        plt.tight_layout()
        plt.yticks([])
        plt.show()
        plt.close()

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "ap": ap,
        "cm": cm,
    }


class BetaModel(nn.Module):
    def __init__(
        self,
        a_positive: float,
        b_positive: float,
        a_negative: float,
        b_negative: float,
    ):
        super().__init__()
        self.a_positive = a_positive
        self.b_positive = b_positive
        self.a_negative = a_negative
        self.b_negative = b_negative
        self.model = get_model(
            a_positive=self.a_positive,
            b_positive=self.b_positive,
            a_negative=self.a_negative,
            b_negative=self.b_negative,
        )

    def forward(
        self,
        labels: torch.LongTensor,
        input_ids: torch.LongTensor | None = None,  # noqa: ARG002
        **kwargs,
    ) -> TokenClassifierOutput:
        """
        Forward pass for the model. Simulates a score based on the input label.

        Args:
            labels: Input labels (0 or 1).
            input_ids: Input IDs (not used in this model - included for compatibility
                with transformers).
            **kwargs: Additional arguments (not used in this model).

        Returns:
            Simulated score based on the beta distribution parameters.
        """
        scores = torch.tensor(self.model(labels.cpu().numpy()))
        # move labels to correct device to enable model parallelism
        loss_fct = nn.BCELoss()
        loss = loss_fct(scores, labels.to(scores.dtype))
        return TokenClassifierOutput(logits=scores, loss=loss)

    @classmethod
    def from_imbalance_and_advantage(
        cls, imbalance: float, model_adv: float
    ) -> "BetaModel":
        """
        Create a BetaModel instance based on the imbalance and model advantage.

        Args:
            imbalance: Fraction of positive labels in the dataset.
            model_adv: Parameterization of model performance.

        Returns:
            A BetaModel instance with the fitted parameters.
        """
        positive_err_rate, negative_err_rate = imbalance_scaled_error_rate(
            model_adv=model_adv, imbalance=imbalance
        )
        a_positive, b_positive, a_negative, b_negative = fit_beta_model_params(
            positive_err_rate=positive_err_rate,
            negative_err_rate=negative_err_rate,
        )
        return cls(
            a_positive=a_positive,
            b_positive=b_positive,
            a_negative=a_negative,
            b_negative=b_negative,
        )


def fit_beta_model_params(
    positive_err_rate: float,
    negative_err_rate: float,
) -> tuple[float, float, float, float]:
    """
    Fit the model parameters based on the error rates.

    Args:
        positive_err_rate: Error rate for the positive class.
        negative_err_rate: Error rate for the negative class.

    Returns:
        Fitted parameters for the model: a_positive, b_positive, a_negative, b_negative.
    """
    a_positive, b_positive = fit_positive_distribution(positive_err_rate)
    a_negative, b_negative = fit_negative_distribution(negative_err_rate)
    return a_positive, b_positive, a_negative, b_negative


def main():
    parser = argparse.ArgumentParser(description="Simulate a parameterized beta model.")
    parser.add_argument(
        "--imbalance",
        type=float,
        required=True,
        help="Fraction of positive labels",
    )
    parser.add_argument(
        "--model-adv",
        required=True,
        type=float,
        help="Model advantage (parameterization of model performance)",
    )
    args = parser.parse_args()

    imbalance = args.imbalance
    model_adv = args.model_adv
    positive_err_rate, negative_err_rate = imbalance_scaled_error_rate(
        model_adv=model_adv, imbalance=imbalance
    )
    print(
        f"Imbalance: {imbalance}, Model Advantage: {model_adv}\n"
        f"Positive Error Rate: {positive_err_rate:.3f}, "
        f"Negative Error Rate: {negative_err_rate:.3f}"
    )

    a_positive, b_positive, a_negative, b_negative = fit_beta_model_params(
        positive_err_rate=positive_err_rate, negative_err_rate=negative_err_rate
    )

    model = get_model(a_positive, b_positive, a_negative, b_negative)
    get_model_stats(imbalance=imbalance, model=model)


if __name__ == "__main__":
    main()
