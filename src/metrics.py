from typing import (
    List,
    Literal,
    Tuple,
    Optional,
    Union)
import numpy as np
import matplotlib.pyplot as plt
from sklearn.calibration import calibration_curve


def adaptive_calibration_error(
        predictions: List[int],
        confidences: List[float],
        n_bins: int = 20
        ) -> float:
    """Variant of the expected calibration error with adaptive bins, i.e.
    where each bin contains approximately the same number of samples.
    ---------------
    :param predictions: 1d array with the binaries if a prediction was correct/incorrect.
    :param confidences: 1d array with associated confidences.
    :param n_bins: int, number of bins.
    :returns: float, the adaptive calibration error."""

    predictions = np.array(predictions)
    confidences = np.array(confidences)

    binned_correct, binned_conf = calibration_curve(predictions, confidences, n_bins=n_bins, strategy="quantile")
    errors = []
    for correct_bin, conf_bin in zip(binned_correct, binned_conf):
        accuracy = np.mean(correct_bin)
        confidence = np.mean(conf_bin)
        error = accuracy - confidence
        errors.append(np.abs(error))

    ace = np.mean(errors)
    return ace


def weighted_average(grouped_df,
                     value_col: str,
                     weight_col: str = "instance"
                     ) -> float:
    """
    Implements the weighted average over all relations for a given column of
    correctness labels.
    """
    weights = grouped_df[weight_col]
    values = grouped_df[value_col]
    return round((values * weights).sum() / weights.sum(), 6)


def brier_score(predictions: List[int],
                confidences: List[float]
                ) -> float:
    """Function which implements the brier score."""
    bs = np.sum((np.array(predictions) - np.array(confidences))**2)*(1/len(predictions))
    return bs


def plot_calibration_curve(
        title: str,
        predictions: List[List[int]],
        confidences: List[List[float]],
        n_bins: int = 20,
        binning_strategy: Literal["uniform", "quantile"] = "quantile",
        linestyles: List[str] = "dotted",
        markers: Union[List[str], None] = "*",
        labels: Optional[List[str]] = None,
        colors: Optional[List[str]] = None,
        axis: Optional[plt.Axes] = None
        ) -> Tuple[plt.Figure, plt.Axes]:
    """Plots (multiple) calibration curves for bins as obtained from sklearn calibration curve."""
    # Axis setup
    if axis is None:
        fig, ax = plt.subplots(figsize=(8, 6), facecolor="white")
    else:
        ax = axis
        fig = ax.figure

    # Identity line (perfect calibration): f(x)=x
    ax.plot([0, 1], [0, 1], color="black", linestyle="solid", label="_nolegend_")

    # Plot each calibration curve
    for i in range(len(predictions)):
        prob_true, prob_pred = calibration_curve(
            predictions[i], confidences[i], n_bins=n_bins, strategy=binning_strategy
        )
        ax.plot(
            prob_pred,
            prob_true,
            color=colors[i],
            marker=markers[i],
            linestyle=linestyles[i],
            label=labels[i]
        )

    # Styling: limit the axes to 0 and 1
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Average Confidence", labelpad=10)
    ax.set_ylabel("Average Accuracy", labelpad=10)
    ax.set_title(title, pad=15)
    ax.legend(loc="upper left", frameon=False)

    # Styling: make the box visible
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(1)
        spine.set_color("black")

    # Styling of the ticks
    ax.xaxis.set_ticks_position("bottom")
    ax.yaxis.set_ticks_position("left")
    ax.set_facecolor("white")
    ax.grid(False)

    plt.tight_layout()
    return fig, ax
