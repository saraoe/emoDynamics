'''
Smoothing EmoDynamics signals
'''
import numpy as np
import pandas as pd
import os
import sys

#Code from newsFluxus Github. 
path = os.path.join("..", "newsFluxus", "src")
sys.path.append(path)
import news_uncertainty


def main(filename:str, out_path:str, span:int):
    '''
    Reads in a csv with emoDynamics extracted. Smooths the signals
    Writes to csv and returns df
    '''
    df = pd.read_csv(filename)
    df["smoothed_transience"] = news_uncertainty.adaptive_filter(df["transience"], span=span)
    df["smoothed_novelty"] = news_uncertainty.adaptive_filter(df["novelty"], span=span)
    df["smoothed_resonance"] = news_uncertainty.adaptive_filter(df["resonance"], span=span)
    df.to_csv(out_path)
    return df

if __name__ == "__main__":
    filename = os.path.join("..", "idmdl", "tweets_emotion_date.csv")
    out = os.path.join("..", "smoothed", "tweets_emotion_date_smoothed_150.csv")
    df = main(filename, out, 150)
    print(df.head())

