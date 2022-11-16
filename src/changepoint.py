"""
Detecting change points in information signal using ruptures
"""
import numpy as np
import ruptures as rpt
import pandas as pd
import re, os
from typing import List, Optional
import argparse


def detect_change_points(ts: np.ndarray, pen: int):
    """
    detecting change points in a time series using Pelt and cost function rbf

    Args:
        ts (np.ndarray): time series
        pen (int): penalty of the model

    Returns
        List of change point locations
    """
    algo = rpt.Pelt(model="rbf").fit(ts)
    change_locations = algo.predict(pen=pen)
    return change_locations


def get_emotion_distribution(emo: str):
    """
    For transforming the emotion distribution from a str to a list of floats
    """
    emo_list = re.sub("[^A-Za-z0-9\s.]+", "", emo).split(" ")
    emo_list = list(map(lambda x: float(x), emo_list))
    return emo_list


def emotions_dict(df: pd.DataFrame, emotion_col: str, labels: List[str]):
    """
    gives dict with list of probabilities for the different emotions in labels from pd.DataFrame

    Args:
        df (pd.DataFrame): data frame with emotion probabilities
        emotion_col (Optional[str]): name of the col with the emotion probabilities
        labels (Optional[List[str]]): List of the emotion labels

    Returns
        dict: with labels as keys and probabilities as values
    """
    emo_lists = list(map(get_emotion_distribution, list(df[emotion_col])))
    zip_emo_lists = [probs for probs in zip(*emo_lists)]
    return {label: probs for label, probs in zip(labels, zip_emo_lists)}


def write_model_df(
    df: pd.DataFrame,
    change_locations: List[int],
    emotion_col: Optional[str] = "emo_prob",
    labels: Optional[List[str]] = [
        "Glæde/Sindsro",
        "Tillid/Accept",
        "Forventning/Interrese",
        "Overasket/Målløs",
        "Vrede/Irritation",
        "Foragt/Modvilje",
        "Sorg/trist",
        "Frygt/Bekymret",
    ],
    out_path: str = None,
):
    """
    writes df for making the linear models. The df contains columns for the number of changepoint,
    resonance, novelty and transience together with the values for all the labels.

    Args:
        df (pd.Dataframe): dataframe with information signal
        change_locations (List[int]): list of where the change points are (to index the df)
        out_path (str): path for saving the dataframe
        emotion_col (Optional[str]): name of the col with the emotion probabilities. Defaults to 'emo_prob'.
        labels (Optional[List[str]]): List of the labels. If it is not specified the labels from the BERT emotion model us used.

    """
    # define the model df
    cols = ["change_point", "resonance", "novelty", "transience"]
    if labels:
        cols += labels
    model_df = pd.DataFrame(columns=cols)

    for n, (i, j) in enumerate(zip([0] + change_locations[:-1], change_locations)):
        tmp_dict = {"change_point": n}
        tmp_df = df.iloc[i:j]
        # get information signals
        for col in ["date", "resonance", "novelty", "transience"]:
            tmp_dict[col] = tmp_df[col]

        if labels:
            # get emotion values
            emo_dict = emotions_dict(tmp_df, emotion_col, labels)
            # concat with dataframe
            tmp = pd.DataFrame.from_dict({**tmp_dict, **emo_dict})
        else:
            tmp = pd.DataFrame.from_dict({**tmp_dict})

        model_df = pd.concat([model_df, tmp])
    if out_path:
        model_df.to_csv(out_path)
    return model_df


def main(df, out_path, penalty, labels):
    change_locations = detect_change_points(df["resonance"].to_numpy(), penalty)
    print(f"change locations found: {change_locations}")
    write_model_df(
        df=df, change_locations=change_locations, out_path=out_path, labels=labels
    )
    print("DONE")


if __name__ == "__main__":
    # define paths and parameters
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--in_file", type=str, required=True, help="Name the input file."
    )
    parser.add_argument(
        "--penalty",
        type=int,
        required=False,
        default=4,
        help="Penalty for the change point detection.",
    )
    parser.add_argument(
        "--labels",
        type=str,
        required=False,
        nargs="+",
        default=None,
        help="List of labels from the classifier. Optional.",
    )
    args = parser.parse_args()

    in_filepath = os.path.join("idmdl", f"{args.in_file}.csv")
    pen = args.penalty
    labels = args.labels
    out_path = os.path.join("idmdl", "changepoints", f"{args.in_file}_cp_{pen}.csv")
    df = pd.read_csv(in_filepath)

    print(
        f"""Running changepoint.py with
                in_path: {in_filepath}
                out_path: {out_path},
                penalty: {pen},
                labels: {labels}
    """,
        end="\n\n",
    )

    main(df, out_path, pen, labels)
