import os

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, Tuple, List, Union

from lm_pub_quiz import Evaluator, Dataset
from scipy.special import softmax

from src.metrics import adaptive_calibration_error, weighted_average


def reduced_scores(row,
                   reduction: str = "sum",
                   only_answers=False
                   ) -> List[float]:
    """
    Helper function to obtain a sentence-level log-likelihood based on different
    reduction strategies (Sum, Sum (A), Mean, Mean (A)). Scores are obtained using
    LM-PUB-QUIZ.
    ---------------
    :param row: Corresponds to the row of a single instance.
    :param reduction: The reduction strategy used for token log-likelihood reduction.
    :param only_answers: Flag if only the answer tokens should correspond to the sentence-level log-likelihood.
    :returns: List of sentence-level log-likelihoods per instance, one for each answer option.
    """
    # Get scores and indices
    scores = row["pll_scores"]
    sub = row["sub_indices"]
    obj = row["obj_indices"]
    template = row["template_indices"]

    # Combining indices per answer option
    combined = obj if only_answers else [sub[i] + obj[i] + template[i] for i in range(len(scores))]

    sentence_level_log_likelihoods = []
    for i in range(len(scores)):
        scores_to_reduce = [scores[i][j] for j in combined[i]]
        sentence_level_log_likelihood = sum(scores_to_reduce) if reduction == "sum" else np.mean(scores_to_reduce)
        sentence_level_log_likelihoods.append(sentence_level_log_likelihood)

    return sentence_level_log_likelihoods


def final_answer(group: pd.DataFrame,
                 strategy: str
                 ) -> Tuple[Optional[int], bool, bool]:
    """
    Helper function to obtain a model's final answer if multiple templates
    are used. Strategies implemented here: plurality vote, majority vote and
    highest confidence answer. For each strategy returns the predicted index,
    if the voting failed or if there was a tie. Plurality vote with the highest
    confidence answer as fallback strategy in the case of a tie between two ansers.
    No ties for majority voting, only fail or success.
    ---------------
    :param group: For each relation and instance, the models template predictions.
    :param strategy: The final answer selection strategy.
    :returns: Predicted index (final answer), was-fail-flag, was-tie-flag.
    """
    # Plurality Vote
    if strategy == "plurality":
        pred_counts = group["predicted_index"].value_counts()
        max_count = pred_counts.max()
        top_predictions = pred_counts[pred_counts == max_count]

        if max_count == 1:  # all answers are different
            return None, True, False    # no answer, fail, no tie

        if len(top_predictions) == 1:   # one single plural answer
            return top_predictions.index[0], False, False   # answer, no fail, no tie
        # Tie between two answers
        tied = group[group["predicted_index"].isin(top_predictions.index)]
        top = tied.loc[tied["base_conf"].idxmax()]  # use the one with highest confidence
        return top["predicted_index"], False, True  # answer, no fail, tie

    # Majority Vote
    elif strategy == "majority":
        num_templates = group["template"].nunique()
        pred_counts = group["predicted_index"].value_counts()
        for pred, count in pred_counts.items():
            if count / num_templates > 0.5:     # there is an answer achieving the majority
                return pred, False, False       # answer, no fail, no tie
        return None, True, False                # otherwise no anser, fail, no tie

    # Maximum-Confidence (Xiong et al., 2024)
    elif strategy == "max_conf":
        top_row = group.loc[group["base_conf"].idxmax()]
        return top_row["predicted_index"], False, False


def plot_accuracy_vs_metric(results_df,
                            estimator: str = "base_conf",
                            metric: str = "brier_score",
                            title: str = "",
                            axis: Optional[plt.Axes] = None
                            ) -> Tuple[plt.Figure, plt.Axes]:
    """
    Helper function to plot accuracy vs calibration metric (ACE, brier score)
    per model family for a given estimator.
    """
    results_df = results_df[results_df["estimator"] == estimator]

    if axis is None:
        fig, ax = plt.subplots(figsize=(8, 6), facecolor="white")
    else:
        ax = axis
        fig = ax.figure

    # Plot the accuracies against the metrics per family
    for family, group in results_df.groupby("family"):
        group.sort_values("accuracy", inplace=True)
        ax.plot(
            group["accuracy"],
            group[metric],
            marker=".",
            linestyle="dotted",
            label=family
        )

    ax.set_xlabel("Accuracy", labelpad=10)
    ax.set_ylabel(metric.replace("_", " ").title(), labelpad=10)
    ax.set_title(title, pad=15)
    ax.legend(title="", loc="upper right", frameon=False)

    # Make the box around the plot visible
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(1)
        spine.set_color("black")

    # More styling: ticks point outwards, white background, no gridlines
    ax.tick_params(direction="out", length=4, width=1)
    ax.xaxis.set_ticks_position("bottom")
    ax.yaxis.set_ticks_position("left")
    ax.set_facecolor("white")
    ax.grid(False)

    plt.tight_layout()
    return fig, ax


def evaluate_reductions(models: Union[str, List[str]] = "gpt2") -> pd.DataFrame:
    """
    Helper function for our first experiment evaluting which reduction method is most
    effective. For each model and reduction strategy finds the ACE and accuracy. Only the
    first template and (baseline) confidence is used.
    """
    if isinstance(models, str):
        models = [models]

    reductions = [
        "pll_scores_sum",
        "pll_scores_avg",
        "pll_scores_answer_sum",
        "pll_scores_answer_avg",
    ]

    all_summary_rows = []

    for model in models:
        output_path = f"../results/scores/{model}"
        file_path = f"{output_path}/scores.json"
        results = pd.read_json(file_path, orient="records", lines=True)
        model_type = results["model_type"].iloc[0]
        results = results[results["template"] == 0]     # restrict analysis to the first template (base_conf)

        results["pll_scores_sum"] = results.apply(
            lambda row: reduced_scores(row, reduction="sum", only_answers=False),
            axis=1)
        results["pll_scores_avg"] = results.apply(
            lambda row: reduced_scores(row, reduction="mean", only_answers=False),
            axis=1)
        results["pll_scores_answer_sum"] = results.apply(
            lambda row: reduced_scores(row, reduction="sum", only_answers=True),
            axis=1)
        results["pll_scores_answer_avg"] = results.apply(
            lambda row: reduced_scores(row, reduction="mean", only_answers=True),
            axis=1)

        for reduction in reductions:
            results[f"{reduction}_correctly_predicted"] = results.apply(
                lambda row: row["answer_idx"] == np.argmax(row[reduction]),
                axis=1
            )
            results[f"{reduction}_confs"] = results.apply(
                lambda row: np.sort(softmax(row[reduction]))[::-1],
                axis=1
            )
            results[f"{reduction}_base_conf"] = results.apply(
                lambda row: row[f"{reduction}_confs"][0],
                axis=1
            )

            # Accuracy as the weighted average over all relations
            grouped = results.groupby("relation").agg({
                "instance": "count",
                f"{reduction}_correctly_predicted": "mean"
            }).reset_index()
            accuracy = weighted_average(grouped, f"{reduction}_correctly_predicted")
            predictions = results[f"{reduction}_correctly_predicted"].astype(int)
            confidences = results[f"{reduction}_base_conf"].tolist()
            ace = round(adaptive_calibration_error(predictions, confidences), 6)

            all_summary_rows.append({
                "model": model,
                "model_type": model_type,
                "reduction": reduction,
                "accuracy": accuracy,
                "ace": ace
            })

    return pd.DataFrame(all_summary_rows)


def get_model_scores(model_id: str = "openai-community/gpt2",
                     model_type: str = "CLM",
                     path_to_data: str = "../data/BEAR_extended",
                     device: str = "cuda",
                     templates: Union[int, List[int]] = 0,
                     batch_size: int = 32
                     ) -> None:
    """
    Function used to obtain the model score using LM-PUB-QUIZ for specified templates.
    Stores the raw (unreduced) scores per instance and template.
    """
    output_path = f"../results/{model_id.split('/')[-1]}"
    file_path = f"{output_path}/scores.json"

    evaluator = Evaluator.from_model(model=model_id, model_type=model_type, device=device)
    bear = Dataset.from_path(path_to_data)
    results = pd.DataFrame()

    if isinstance(templates, int):
        templates = [templates]
    for template in templates:
        temp_df = (
            evaluator
            .evaluate_dataset(bear, template_index=template, batch_size=batch_size, reduction=None)
            .joined_instance_table()
            .reset_index()
            .assign(template=template)
        )
        results = pd.concat([results, temp_df], ignore_index=True)

    results["model_type"] = model_type
    results["model"] = model_id.split("/")[-1]
    os.makedirs(output_path, exist_ok=True)
    results.to_json(file_path, orient="records", lines=True)
    print(f"Scores saved to: {file_path}")
