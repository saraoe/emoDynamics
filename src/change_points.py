"""
Script for extracting change points in signal using ruptures
"""

import ruptures as rpt
import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt
import os


def get_change_points(signal, penalty:int=4, model:str="rbf"):
    mdl = rpt.Pelt(model=model).fit(np.array(signal))
    change_points = mdl.predict(pen=penalty)
    return change_points

def plot_change_points(time, signal, shifts, title="Change points in the signal"):
    time = pd.to_datetime(time)
    plt.figure(figsize=(16,4))
    plt.plot(time, signal)
    for x in shifts:
        plt.axvline(time[x-1], lw=2, color='red')
    plt.title(title)

def get_date_from_change_point(timeseries, change_points):
    dates = [timeseries[i-1] for i in change_points]
    return dates

if __name__ == "__main__":
    path = os.path.join("..", "src", "idmdl", "tweets_emo_date_W3.csv")
    df = pd.read_csv(path)

    change_points = get_change_points(df["resonance"])
    dates = get_date_from_change_point(df["date"], change_points)

    for date in dates:
        print(f'Change in signal happened on {date}')
