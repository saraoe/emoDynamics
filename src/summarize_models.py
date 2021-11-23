'''
For summarizing the BERT emotion probabilities
'''
import pandas as pd
import numpy as np
import re
import ndjson
import time


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
    return get_emotion_distribution(emo, 3)


def emotion_distribution_mean(emo_lists: list) -> list:
    '''
    Takes mean of each emotion probability in a BERT emotion probability list.
    '''
    n = len(emo_lists)
    return [(np.mean(prob), np.std(prob)) for prob in zip(*emo_lists)]


def read_in_csv(filepath: str, time_col: str, emo_col: str, tweets=True, fix_col=False):
    '''
    Function for reading in the csv with emotion BERT scores

    Args:
        filepath (str): path for the csv file
        time_col (str): column in df with time (e.g. 'created_at')
        emo_col (str): column in df with the emotion probabilities
        tweets (bool): True if tweets, False if newspapers
        fix_col (bool): if column names needs to be fixed (for the 2019 tweets)
    
    return
        pandas.DataFrame
    '''
    ## load in data ##
    print('read data')
    chunks = pd.read_csv(filepath, header = 0,
                        chunksize = 1000)

    df = pd.DataFrame()
    for i, chunk in enumerate(chunks):
        if fix_col:
            col_names = list(chunk.columns[1:]) + ['None']
            col_dict = {old: new for (old, new) in zip(chunk.columns, col_names)}
            chunk = chunk.rename(columns = col_dict)

        # chunk = chunk[chunk['Bert_emo_laden'] == 'Emotional'] # only include emotional laden tweets
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


def write_ndjson_by_group(df, group_by: list, filename: str, emo_col: str):
    '''
    Groups df by arguments in group_by list. 
    Writes ndjson with group and emotion distribution
    '''
    grouped = df.groupby(group_by)
    for name, group in grouped:
        print(name, ', time =', time.time()-start_time)
        if emo_col == 'Bert_emo_emotion_prob':
            emo_lists = list(map(get_emotion_distribution,list(group[emo_col])))
        if emo_col == 'polarity_prob':
            emo_lists = list(map(get_polarity_distribution,list(group[emo_col])))
        emo_prob, emo_prob_sd = zip(*emotion_distribution_mean(emo_lists))
        line = [{'group': name, 'emo_prob': emo_prob, 'emo_prob_sd': emo_prob_sd}]
        with open(f'{filename}.ndjson', 'a') as f:
            ndjson.dump(line, f)
            f.write('\n')


if __name__ == '__main__':
    start_time = time.time()
    df = read_in_csv('/home/commando/stine-sara/data/emotion_tweets_2019.csv', 
                     time_col = 'created_at', emo_col = 'polarity_prob')#, fix_col=True)

    # write ndjson
    write_ndjson_by_group(df, group_by = ['date', 'hour'], 
                          filename = "../summarized_emo/tweets19_pol_date_hour_sd", 
                          emo_col = 'polarity_prob')
    print('finished grouped by date and hour')

    write_ndjson_by_group(df, group_by = ['date'], 
                          filename = "../summarized_emo/tweets19_pol_date_sd", 
                          emo_col = 'polarity_prob')
    print('finished grouped by date')
    
    
    
