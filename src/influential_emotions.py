'''
Analyzing and investigating what emotions seem to be driving the effect
'''
import os
import pandas as pd
from sklearn.linear_model import LinearRegression
import numpy as np
import matplotlib.pyplot as plt

def main_by_date(path:str, main_col:str, date_col:str="created_at", new_main_name:str="main_emotion"):
    """This function extracts the main emotion used in tweets for each day.

    Args:
        path (str): path to where csv with main emotion of each day is stored
        main_col (str): column name that indicates the main emotion of the day
        date_col (str, optional): column name of date column. Defaults to "created_at".
        new_main_name (str, optional): New column name for the column indicating main emotion. Defaults to "main_emotion".

    Returns:
        combined (pd.DataFrame): dataframe with the main emotion of each day along with the proportion of tweets being that emotion
    """    
    df = pd.read_csv(path)
    df["date"] = pd.to_datetime(df[date_col], utc=True).dt.strftime('%Y-%m-%d')

    # Count number of tweets for each emotion for each day
    count = pd.DataFrame({'n_tweets_main' : df.groupby(['date', main_col]).size()}).reset_index()
    idx = count.groupby(["date"])["n_tweets_main"].transform(max) == count['n_tweets_main']
    count = count[idx]

    # Count total number of tweets each day
    total_tweets = pd.DataFrame({'n_tweets_total' : df.groupby('date').size()}).reset_index()
    
    # Combine, calculate proportion and rename
    combined = count.merge(total_tweets)
    combined["main_proportion"] = combined["n_tweets_main"]/combined["n_tweets_total"]
    combined = combined.rename(columns = {main_col: new_main_name})

    return combined


def emotion_by_date(df:pd.DataFrame, main_col:str, date_col:str="created_at"):
    """Calculate the number of tweets in each emotion category each day

    Args:
        df (pd.DataFrame): dataframe with date column (specified by date_col) and emotion column (main_col)
        main_col (str): column name indicating where emotion categories are specified
        date_col (str, optional): column name of date column. Defaults to "created_at".

    Returns:
        count (pd.DataFrame): dataframe with count and proportion of tweets for all emotions in main_col
    """    
    df["date"] = pd.to_datetime(df[date_col], utc=True).dt.strftime('%Y-%m-%d')

    # Count number of tweets for each emotion for each day
    count = pd.DataFrame({'n_tweets' : df.groupby(['date', main_col]).size()}).reset_index()
    # Count total number of tweets each day
    totals = count[["date", "n_tweets"]].groupby(["date"]).sum("n_tweets").reset_index()
    # Combine
    totals.columns = ["date", "total"]
    count = count.merge(totals, on=["date"])
    
    count["proportion"] = count["n_tweets"]/count["total"]

    return count


if __name__ == "__main__":
    # Load the csv not summarized
    years = ["2019", "2021"]
    for year in years:
        print(f"Starting summarization for file emotion_tweets_{year}")
        path = os.path.join("/home", "commando", "stine-sara", "data", f"emotion_tweets_{year}.csv")
        df = pd.read_csv(path)
        print(f"Head of emotion_tweets_{year}:")
        print(df.head())
        
        col_name = "Bert_emo_emotion"
        df = emotion_by_date(df[["created_at", col_name]], col_name)
        print(f"Summarized file emotion_tweets_{year}, now saving")
        df.to_csv(f"summarized_emo/emotion_proportions_{year}.csv")

    # emo_df = pd.read_csv("summarized_emo/main_emo_day.csv")
    # dynamic_df = pd.read_csv("idmdl/tweets_emo_date_W3.csv")
    # df = emo_df.merge(dynamic_df)
    # df = df[["date", "main_emotion", "main_proportion", "novelty", "resonance"]]
    # df = df.sort_values("main_proportion")

    # # Linear regression
    # measure = "novelty"
    # X = np.array(df["main_proportion"]).reshape(-1,1)
    # y = np.array(df[measure])
    # reg = LinearRegression().fit(X, y)
    # performance = reg.score(X, y)

    # print(df.head())
    # print(performance)

    # fig, ax = plt.subplots()
    # ax.plot(df["main_proportion"],df[measure])
    # ax.set_xlabel("Proportion of tweets being the main emotion")
    # ax.set_ylabel(measure.capitalize())
    # fig.suptitle(f"{measure.capitalize()} predicted by proportion of tweets being main emotion")
    # fig.show()