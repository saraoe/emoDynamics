'''
Calculating corelation between signals
'''
from numpy.core.fromnumeric import var
import pandas as pd
import os
import matplotlib.pyplot as plt; 
import numpy as np
from scipy.stats import zscore
# import matplotlib; matplotlib.use('TkAgg')

def calculate_correlation(data: pd.DataFrame, rolling: bool=False, w: int=None):
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
    data_int = data.interpolate() # Deal with NaN values using interpolation
    if rolling: 
        rolling_r = data_int.iloc[:,0].rolling(window=w, center=False).corr(data_int.iloc[:,1])
        return rolling_r

    else:
        pearson_r = data_int.iloc[:,0].corr(data_int.iloc[:,1])
        return pearson_r

def plot_rolling_correlations(data: pd.DataFrame, corr_var: pd.Series=None, w :int=None, measure: str="resonance"):
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

    f,ax = plt.subplots(2,1,figsize=(14,6),sharex=True)
    ax[0].plot(data.iloc[:,0], label=data.columns[0])
    ax[0].plot(data.iloc[:,1], label=data.columns[1])
    ax[0].legend()
    ax[1].plot(corr_var, label="Pearson correlation")
    ax[0].set(xlabel='Frame', ylabel=measure)
    ax[1].set(xlabel='Frame', ylabel='Pearson r')
    plt.suptitle(f'{measure} data and rolling window (size = {w}) correlation')
    return f

    

if __name__ == "__main__":
    emotion = os.path.join("..", "idmdl", "tweets_recol_emotion_date.csv")
    polarity = os.path.join("..", "idmdl", "tweets_recol_polarity_date.csv")
    emo_df = pd.read_csv(emotion)
    pol_df = pd.read_csv(polarity)
    w=3

    # Combining to dataframe
    df = pd.DataFrame({"emotion_resonance":emo_df["resonance"], 
                       "polarity_resonance":pol_df["resonance"]})
    df_scaled = pd.DataFrame({"emotion_resonance":zscore(emo_df["resonance"]), 
                       "polarity_resonance":zscore(pol_df["resonance"])})

    # Calculate rolling correlation coefficients
    rolling_scaled = calculate_correlation(df_scaled, rolling=True, w=w)
    rolling = calculate_correlation(df, rolling=True, w=w)

    # Calculate overall correlation coefficient
    coef = calculate_correlation(df)
    coef_scaled = calculate_correlation(df_scaled)

    print(f'Overall correlation coefficient between signals: {coef}')
    print(f'Overall correlation coefficient between scaled signals: {coef_scaled}')

