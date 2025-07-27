import json

import numpy as np
import pandas as pd
from scipy.special import softmax

from src.utils import final_answer, reduced_scores


def base_confidence(instances_df: pd.DataFrame) -> pd.Series:
    """Function to obtain the Base-Confidence (Ploner et al., 2024) as
    defined in Equation C.1. Any template can be used for this estimate.
    ---------------
    :param instances_df: Dataframe with the instance-level results per template.
    :returns: A pandas series of the Base-Confidence."""
    return instances_df["conf_scores"].apply(lambda x: np.max(x))


def margin_confidence(instances_df: pd.DataFrame) -> pd.Series:
    """Function to obtain the Confidence-Margin (Ploner et al., 2024) as
    defined in Equation C.2. Any template can be used for this estimate.
    ---------------
    :param instances_df: The dataframe with the instance-level results per template.
    :returns: A pandas series of the confidence margin."""
    return instances_df["conf_scores"].apply(lambda x: x[0] - x[1])


def average_confidence(instances_df: pd.DataFrame,
                       num_templates: int = 5
                       ) -> pd.DataFrame:
    """Function to obtain Average-Confidence as defined in Equation C.3
    for both, plurality and majority voting using multiple templates.
    Also counts the number of failures/ties for each strategy.
    ---------------
    :param instances_df: Dataframe with instance-level results per template
    :param num_templates: The total number of template available.
    :returns: Dataframe with average confidence per instance
    """
    strategies = ["plurality", "majority"]
    instances_df = (
        instances_df[(instances_df["template"] != 5) &
                     (instances_df["template"] != 6)]
    )     # remove predictions from epistemic templates
    grouped = instances_df.groupby(["relation", "instance"])
    records = []

    for (relation, instance), template in grouped:
        ground_truth = template["answer_idx"].iloc[0]   # ground truth is the same for all templates
        bear_score = template["correctly_predicted"].mean()     # bear score (for comparison later)
        record = {
            "relation": relation,
            "instance": instance,
            "bear_score": bear_score
        }

        for strategy in strategies:
            pred_idx, fail, tie = final_answer(template, strategy)

            if pred_idx is None:
                avg_conf = 0.0      # we interpret vote fails as refusals and hard-code to 0 confidence
                correct = False     # hard-coded false with 0 confidence yields 0 calibration error
            else:
                matching = template[template["predicted_index"] == pred_idx]    # matches correspond to confidence
                avg_conf = matching["base_conf"].sum() / num_templates
                correct = (pred_idx == ground_truth)

            # Per-instance confidence, correct-flag, fail-flag and tie-flag
            record[f"average_conf_{strategy}"] = avg_conf
            record[f"{strategy}_correct"] = correct
            record[f"{strategy}_fail"] = int(fail)
            record[f"{strategy}_tie"] = int(tie)

        records.append(record)

    return pd.DataFrame(records)


def consistency_confidence(instances_df: pd.DataFrame,
                           num_templates: int = 5
                           ) -> pd.DataFrame:
    """Computes the consistency confidence for plurality and majority voting
    as defined in Equation C.4.
    ---------------
    :param instances_df: Dataframe with instance-level results per template.
    :param num_templates: The total number of template available.
    :returns: DataFrame with consistency confidence per instance.
    """
    strategies = ["plurality", "majority"]
    instances_df = (
        instances_df[(instances_df["template"] != 5) &
                     (instances_df["template"] != 6)])  # remove predictions from epistemic templates
    grouped = instances_df.groupby(["relation", "instance"])
    records = []

    for (relation, instance), template in grouped:
        record = {"relation": relation, "instance": instance}

        for strategy in strategies:
            pred_idx, _, _ = final_answer(template, strategy)   # ties/fail-flags already obtained above

            if pred_idx is None:
                consistency = 0.0       # same for the correct-flags
            else:
                vote_count = (template["predicted_index"] == pred_idx).sum()    # number of matches
                consistency = vote_count / num_templates

            record[f"consistency_conf_{strategy}"] = consistency

        records.append(record)

    return pd.DataFrame(records)


def weighted_confidence(instances_df: pd.DataFrame) -> pd.DataFrame:
    """Function to derive the Weighted-Confidence from Xiong et al., (2024)
    as defined in Equation C.5.
    ---------------
    :param instances_df: The dataframe with the instance-level results per template.
    :returns: The grouped dataframe with one (final) prediction and confidence per instance.
    """
    instances_df = instances_df[(instances_df["template"] != 5) & (instances_df["template"] != 6)]
    records = []

    for (relation, instance), template in instances_df.groupby(["relation", "instance"]):
        ground_truth = template["answer_idx"].iloc[0]
        pred_idx, _, _ = final_answer(template, strategy="max_conf")
        total_conf = template["base_conf"].sum()

        matching_conf = template[template["predicted_index"] == pred_idx]["base_conf"].sum()
        weighted_conf = matching_conf / total_conf
        correct = (pred_idx == ground_truth)

        records.append({
            "relation": relation,
            "instance": instance,
            "weighted_conf": weighted_conf,
            "max_conf_correct": correct
        })

    return pd.DataFrame(records)


def marker_confidence(instances_df: pd.DataFrame) -> pd.DataFrame:
    """Function to obtain a models prediction and associated base-confidences
    from epistemic marker injected templates (template indices 5 and 6).
    ---------------
    :param instances_df: DataFrame with instance-level results per template.
    :returns: Dataframe with the epistemic results."""
    # templates 6 and 7 in the (combined) metadata are the injected ones
    weakener_df = instances_df[instances_df["template"] == 5].copy()
    strengthener_df = instances_df[instances_df["template"] == 6].copy()

    weakener_df.loc[:, "weakener_conf"] = base_confidence(weakener_df)
    strengthener_df.loc[:, "strengthener_conf"] = base_confidence(strengthener_df)

    weakener_df.loc[:, "weakener_correctly_predicted"] = weakener_df.apply(
        lambda row: np.argmax(row["pll_scores"]) == row["answer_idx"],
        axis=1
    )
    strengthener_df.loc[:, "strengthener_correctly_predicted"] = strengthener_df.apply(
        lambda row: np.argmax(row["pll_scores"]) == row["answer_idx"],
        axis=1
    )

    marker_results = weakener_df.merge(strengthener_df, on=["relation", "instance"])
    cols_to_keep = ["relation", "instance", "weakener_conf", "strengthener_conf",
                    "weakener_correctly_predicted", "strengthener_correctly_predicted"]

    return marker_results[cols_to_keep]


def get_confidence_estimates(model: str = "gpt2",
                             reduction: str = "sum",
                             only_answers: bool = False
                             ) -> pd.DataFrame:
    """Function to obtain the instance-level results: confidence estimates,
    correctness-labels etc. used for our analysis later.
    ---------------
    :param model: The model to evaluate.
    :param reduction: The reduction method for the token log-likelihoods.
    :param only_answers: Flag if only the answer tokens should correspond to the sentence-level log-likelihood.
    :returns: The dataframe with the final instance-level results for model."""

    # Load the raw instance-level results
    results = pd.read_json(f"../results/scores/{model}/scores.json",
                           orient="records", lines=True)

    # Sentence-level log-likelihood per instance and answer option
    results["pll_scores"] = results.apply(
        lambda row: reduced_scores(row, reduction=reduction, only_answers=only_answers),
        axis=1
    )

    # For each instance and template we obtain predicted index, correctness and normalized softmax scores
    results["predicted_index"] = results["pll_scores"].apply(np.argmax)
    results["correctly_predicted"] = results.apply(
        lambda row: row["answer_idx"] == row["predicted_index"], axis=1
    )
    results["conf_scores"] = results["pll_scores"].apply(
        lambda scores: np.sort(softmax(scores))[::-1]
    )

    # Confidence estimates which rely on a single template
    results["base_conf"] = base_confidence(results)
    results["margin_conf"] = margin_confidence(results)

    # Confidence estimates which rely on multiple templates
    weighted_df = weighted_confidence(results)
    average_df = average_confidence(results)
    consistency_df = consistency_confidence(results)

    # Confidence estimate which use the injected templates
    marker_df = marker_confidence(results)

    # Merge results: confidence scores and correctness per instance for the different approaches
    summary_df = (
        results
        .merge(weighted_df, on=["relation", "instance"])
        .merge(average_df, on=["relation", "instance"])
        .merge(consistency_df, on=["relation", "instance"])
        .merge(marker_df, on=["relation", "instance"])
        .query("template == 0")     # keep results from first template for single template estimates
        .drop(columns=["template", "tokens", "sub_indices",
                       "obj_indices", "template_indices", "conf_scores"])
        .assign(model=model)
    )

    # Load and merge domain metadata
    with open("../data/relation_info.json") as f:
        domain_data = json.load(f)
    domain_df = pd.DataFrame.from_dict(domain_data, orient="index").reset_index()
    domain_df.columns = ["relation", "domains"]

    return summary_df.merge(domain_df, on="relation", how="left")
