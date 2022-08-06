"""
Calculating corelation between signals
"""
import pandas as pd
import os
import matplotlib.pyplot as plt
import argparse
from scipy.stats import zscore

# import matplotlib; matplotlib.use('TkAgg')


def calculate_correlation(data: pd.DataFrame, rolling: bool = False, w: int = None):
    """This functions calculates Pearson correlation coefficients between two pandas Series.

    Args:
        data (pd.DataFrame): DataFrame with two columns to calculate correlation between
        rolling (bool, optional): If True, correlation in rolling windows is calculated. Defaults to False.
                                  If True, w is required
        w (int, optional): Window size to use for rolling window. Defaults to None.

    Returns:
        pd.Series or np.float: variable with rolling correlation coefficients
                               or overall correlation coefficient
    """
    data_int = data.interpolate()  # Deal with NaN values using interpolation
    if rolling:
        rolling_r = (
            data_int.iloc[:, 0]
            .rolling(window=w, center=False)
            .corr(data_int.iloc[:, 1])
        )
        return rolling_r

    else:
        pearson_r = data_int.iloc[:, 0].corr(data_int.iloc[:, 1])
        return pearson_r


def plot_rolling_correlations(
    data: pd.DataFrame,
    filename: str,
    corr_var: pd.Series = None,
    w: int = None,
    measure: str = "resonance",
):
    """This function plots rolling window correlation coefficients along with original data

    Args:
        data (pd.DataFrame): DataFrame with the two columns correlation was calculated between
        corr_var (pd.Series, optional): Variable containing rolling window correlations. Defaults to None
                                        If not provided, then it is calculated from data.
        w (int): Window to use for rolling plots
        measure (str, optional): Measure correlation is calculated between. Defaults to "resonance".
    """
    if not corr_var:
        corr_var = calculate_correlation(data, rolling=True, w=w)

    f, ax = plt.subplots(2, 1, figsize=(14, 6), sharex=True)
    ax[0].plot(data.iloc[:, 0], label=data.columns[0])
    ax[0].plot(data.iloc[:, 1], label=data.columns[1])
    ax[0].legend()
    ax[1].plot(corr_var, label="Pearson correlation")
    ax[0].set(xlabel="Frame", ylabel=measure)
    ax[1].set(xlabel="Frame", ylabel="Pearson r")
    plt.suptitle(f"{measure} data and rolling window (size = {w}) correlation")
    plt.savefig(os.path.join("fig", f"{filename}.png"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--files",
        type=str,
        required=True,
        nargs=2,
        help="Filenames for the csv files with measures to be correlated.",
    )
    parser.add_argument(
        "--measure",
        type=str,
        required=True,
        help="Name of the measure that should be correlated.",
    )
    parser.add_argument(
        "--window",
        type=int,
        required=False,
        default=3,
        help="Window for rolling average. Default is 3.",
    )
    parser.add_argument(
        "--plots",
        type=str,
        required=False,
        default=None,
        help="Whether you want plots or not. If you do specify filename for plots. Default is None.",
    )
    parser.add_argument(
        "--rootpath",
        type=str,
        required=False,
        default="idmdl",
        help="Rootpath for the files. Default is 'idmdl'.",
    )
    args = parser.parse_args()

    files = args.files
    rootpath = args.rootpath
    df1 = pd.read_csv(os.path.join(rootpath, f"{files[0]}.csv"))
    df2 = pd.read_csv(os.path.join(rootpath, f"{files[1]}.csv"))
    measure = args.measure
    w = args.window
    filename = args.plots

    # match dates
    df1 = df1[df1["date"].isin(df2["date"])].reset_index(drop=True)
    df2 = df2[df2["date"].isin(df1["date"])].reset_index(drop=True)

    # Combining to dataframe
    df = pd.DataFrame(
        {
            "df1_measure": df1[measure],
            "df2_measure": df2[measure],
        }
    )
    df_scaled = pd.DataFrame(
        {
            "df1_measure": zscore(df1[measure]),
            "df2_measure": zscore(df2[measure]),
        }
    )

    # Calculate rolling correlation coefficients
    rolling_scaled = calculate_correlation(df_scaled, rolling=True, w=w)
    rolling = calculate_correlation(df, rolling=True, w=w)

    # Calculate overall correlation coefficient
    coef = calculate_correlation(df)
    coef_scaled = calculate_correlation(df_scaled)

    print(f"Overall correlation coefficient between signals: {coef}")
    print(f"Overall correlation coefficient between scaled signals: {coef_scaled}")

    if filename:  # plots
        plot_rolling_correlations(
            data=df,
            w=w,
            measure=measure,
            filename=filename,
        )
        plot_rolling_correlations(
            data=df_scaled,
            w=w,
            measure=measure,
            filename=f"{filename}_scaled",
        )
