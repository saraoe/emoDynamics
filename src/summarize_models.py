'''
For summarizing the BERT emotion probabilities
'''
import argparse
import pandas as pd
import numpy as np
import re
import ndjson
import time
import os
from typing import List


## Define functions ##
def get_emotion_distribution(emo: str, n: int=8):
    '''
    For transforming the BERT emotion distribution from a str to a list of floats
    If there is no emotion distribution, it returns NaN
    '''
    if not isinstance(emo, str): # if emo == NaN
        return emo
    emo_list = re.split(r'\s+', emo[1:-1])[:n]
    emo_list = list(map(lambda x: float(x), emo_list))
    return emo_list


def get_polarity_distribution(emo: str):
    '''
    Using get_emotion_distribution on polarity, where there are only
    three categories
    '''
    return get_emotion_distribution(emo, 3)


def emotion_distribution_mean(emo_lists: list) -> list:
    '''
    Takes mean of each emotion probability in a BERT emotion probability list.
    '''
    return [(np.mean(prob), np.std(prob)) for prob in zip(*emo_lists)]


def read_in_csv(filepath: str, time_col: str, emo_col: str, tweets=True, only_emo=False):
    '''
    Function for reading in the csv with emotion BERT scores

    Args:
        filepath (str): path for the csv file
        time_col (str): column in df with time (e.g. 'created_at')
        emo_col (str): column in df with the emotion probabilities
        tweets (bool): True if tweets, False if newspapers
        only_emo (bool): whether only emotional tweets should be included
    
    return
        pandas.DataFrame
    '''
    start_time = time.time()
    ## load in data ##
    print('read data')
    chunks = pd.read_csv(filepath, header = 0,
                        chunksize = 1000)

    df = pd.DataFrame()
    for i, chunk in enumerate(chunks):
        if only_emo:
            chunk = chunk[chunk['Bert_emo_laden'] == 'Emotional'] # only include emotional laden tweets
        chunk = chunk[[time_col, emo_col]] # only include certain columns
        
        df = pd.concat([df,chunk])
        if i % 10 == 0:
            print('at chunk ', i)

    print('finished reading data. Time = ', time.time()-start_time)
    
    
    # add date and hour
    df["date"] = pd.to_datetime(df[time_col], utc=True).dt.strftime('%Y-%m-%d')
    if tweets:
        df["hour"] = pd.to_datetime(df[time_col], utc=True).dt.strftime('%H')
    return df


def write_ndjson_by_group(df: pd.DataFrame, group_by: List[str], filename: str, emo_col: str):
    '''
    Groups df by arguments in group_by list. 
    Writes ndjson with group and emotion distribution

    Args
        df (pandas.DataFrame): Dataframe with the data
        group_by (List[str]): List of column to group by (e.g. date)
        filename (str): Name of the file to be written
        emo_col (str): Name of the column with the emotion distribution
    
    return
        None
    '''
    grouped = df.groupby(group_by)
    for name, group in grouped:
        print('Group', name)
        if emo_col == 'Bert_emo_emotion_prob':
            emo_lists = list(map(get_emotion_distribution,list(group[emo_col])))
        if emo_col == 'polarity_prob':
            emo_lists = list(map(get_polarity_distribution,list(group[emo_col])))
        emo_prob, emo_prob_sd = zip(*emotion_distribution_mean(emo_lists))
        line = [{'group': name, 'emo_prob': emo_prob, 'emo_prob_sd': emo_prob_sd}]
        with open(f'{filename}.ndjson', 'a') as f:
            ndjson.dump(line, f)
            f.write('\n')


def main(filepath: str, output_name: str, emo_col: str, time_col: str, only_emo: bool):
    df = read_in_csv(filepath, 
                     time_col = time_col, 
                     emo_col = emo_col,
                     only_emo = only_emo)

    # write ndjson
    write_ndjson_by_group(df, group_by = ['date', 'hour'], 
                          filename = os.path.join('..', 'summarized_emo', f'{output_name}_date_hour'), 
                          emo_col = emo_col)
    print('finished grouped by date and hour')

    write_ndjson_by_group(df, group_by = ['date'], 
                          filename = os.path.join('..', 'summarized_emo', f'{output_name}_date'), 
                          emo_col = emo_col)
    print('finished grouped by date')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--filepath', type=str, required=True,
                        help='Path for the file containing the emotion scores')
    parser.add_argument('--output_name', type=str, required=True,
                        help='Name of the output file')
    parser.add_argument('--emotion_col', type=str, required=True,
                        help='The name of the column with the emotion scores')
    parser.add_argument('--time_col', type=str, required=True,
                        help='The name of the column with time/date')
    parser.add_argument('--only_emo', type=bool, required=False, default=False,
                        help='whether only emotional tweets should be included')
    args = parser.parse_args()

    print(f'''Running summarize_models.py with:
             filepath={args.filepath},
             output_name={args.output_name},
             emo_col={args.emotion_col},
             time_col={args.time_col},
             only_emo={args.only_emo}''')
    main(filepath=args.filepath,
         output_name=args.output_name,
         emo_col=args.emotion_col,
         time_col=args.time_col,
         only_emo=args.only_emo)
