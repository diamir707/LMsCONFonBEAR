from typing import List, Tuple, Literal
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.isotonic import IsotonicRegression
from sklearn.calibration import calibration_curve
from src.metrics import adaptive_calibration_error, brier_score


def make_train_test_set(
        model_data: pd.DataFrame,
        target: str = "correctly_predicted",
        top_label_softmax: str = "base_conf",
        test_size: float = 0.3,
        random_state: int = 444
        ) -> Tuple[pd.DataFrame, ...]:
    """
    Function to create the design matrix X and target y for regression/histogram binning.
    ---------------
    :param model_data: the dataframe with all model scores.
    :param target: the target variable.
    :param top_label_softmax: binary indicating the model was correct/false.
    :param test_size: size of the test set.
    :param random_state: random state used for splitting the sets.
    :returns: the splits of the dataset.
    """
    data = model_data.copy()

    x_mat = data[[top_label_softmax, "margin_conf", "model"]]
    y = data[target].astype(int)

    return train_test_split(
        x_mat,
        y,
        test_size=test_size,
        random_state=random_state,
    )


def histogram_binning(x_train: List[float],
                      y_train: List[int],
                      x_test: List[float],
                      n_bins: int = 20
                      ) -> List[float]:
    """
    Function which implements histogram binning using quantile binnign.
    Modified code of https://github.com/ethen8181/machine-learning.git and
    sklearn source code from calibration_curve.
    ---------------
    :param x_train: Uncalibrated scores for training.
    :param y_train: Correspond fractions of correct predictions.
    :param x_test: Uncalibrated scores of the test set.
    :param n_bins: Number of bins used for quantile binning.
    :returns: The list with the calibrated (test) confidence scores.
    """
    prob_true, prob_pred = calibration_curve(
        y_train, x_train, n_bins=n_bins, strategy="quantile"
    )

    # Calculate quantile-based bin edges on training set
    quantiles = np.linspace(0, 1, n_bins+1)
    bins = np.percentile(x_train, quantiles*100)

    # Given the bin edges fined the bins on the test set
    binids = np.searchsorted(bins[1:-1], x_test)
    calibrated_confidence = [prob_true[i] for i in binids]

    return calibrated_confidence


def recalibrate_lms(
        data: pd.DataFrame,
        model_names: List[str],
        method: Literal["isotonic", "histogram"] = "isotonic"
        ) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Recalibrates the confidence score for each LM using histogram binning and
    isotonic regression. Evaluates the recalibrated scores in terms of ACE and BS.
    ---------------
    :param data: Long pandas dataframe with all results.
    :param model_names: The list of models to recalibrate.
    :param method: str, recalibration method.
    :returns: Two pandas dataframes, one with the recalibrated predictions, the other with the metrics.
    """
    predictions_list = []
    metrics_list = []

    for model_name in model_names:
        model_data = data[data["model"] == model_name]
        x_train, x_test, y_train, y_test = make_train_test_set(model_data)
        # Fit calibration model
        if method == "isotonic":
            cal_model = IsotonicRegression(increasing=True, out_of_bounds="clip")
            cal_model.fit(x_train["base_conf"], y_train)
            calibrated_conf = cal_model.predict(x_test["base_conf"])
        else:
            calibrated_conf = histogram_binning(x_train["base_conf"], y_train.values.tolist(), x_test["base_conf"])

        # Dataframe with the predictions
        pred_df = pd.DataFrame({
            "model": model_name,
            "true_label": y_test.values,
            "uncalibrated_confidence": x_test["base_conf"],
            "calibrated_confidence": calibrated_conf,
            "margin_confidence": x_test["margin_conf"],
            "method": method
        })
        predictions_list.append(pred_df)

        # Evaluation Metrics on the test set for method and confidence margin
        ace = adaptive_calibration_error(y_test.values.tolist(), calibrated_conf)
        ace_margin = adaptive_calibration_error(y_test.values.tolist(), x_test["margin_conf"])
        brier = brier_score(y_test.values.tolist(), calibrated_conf)
        brier_margin = brier_score(y_test.values.tolist(), x_test["margin_conf"])

        metrics_list.append({
            "model": model_name,
            "brier_score_margin": brier_margin,
            "brier_score": brier,
            "ACE": ace,
            "ACE_margin": ace_margin,
            "method": method
        })

    predictions_df = pd.concat(predictions_list, ignore_index=True)
    metrics_df = pd.DataFrame(metrics_list)

    return predictions_df, metrics_df
