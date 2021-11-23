'''
Applying the methods of newsFluxus to BERT emotion distribution
- Novelty
- Transience
- Resonance 

Emotion distributions from tweets and news frontpages
'''
import os, sys
import ndjson, datetime
import pandas as pd
path = os.path.join("..", "newsFluxus", "src")
sys.path.append(path)

from main_extractor import extract_novelty_resonance

def get_emo(d: dict) -> list:
    '''
    returns BERT emotion distribution
    '''
    return d['emo_prob']


def get_time(d: dict) -> datetime.datetime:
    '''
    returns time as type datetime.datetime
    '''
    group = d['group']
    if isinstance(group, list): # when both date and time is included
        date, hour = group
        date = list(map(int, date.split('-')))
        hour = int(hour)
        return datetime.datetime(*date, hour)
    time = list(map(int, group.split('-')))
    return datetime.datetime(*time)


def get_data_time(d: dict) -> list:
    '''
    takes dictionary and returns data and time as a tuple
    
    data is a list of emotion distributions
    time is the corresponding timepoints as datetime
    '''
    data = list(map(get_emo, d))
    time = list(map(get_time, d))
    return data, time


def extract_excluded_emos(filename: str, out_folder: str, window: int, labels: list = [
        "Glæde/Sindsro",
        "Tillid/Accept",
        "Forventning/Interrese",
        "Overasket/Målløs",
        "Vrede/Irritation",
        "Foragt/Modvilje",
        "Sorg/trist",
        "Frygt/Bekymret",
    ]):
    '''
    Extracts novelty, transience and resonance excluding each emotion one at a time
    Writes to csv
    '''
    with open(filename) as f:
        emo_file = ndjson.load(f)
    
    data, time = get_data_time(emo_file)

    #Leave one emotion out at a time and extract novelty and resonance
    for i in range(len(labels)):
        df = pd.DataFrame()
        df['date'] = time
        df[f'no_{labels[i]}'] = [d[:i] + d[i+1:] for d in data]
        out_path = out_folder + f"tweets_emotion_no_{labels[i][:4]}.csv"
        df = extract_novelty_resonance(df, df[f'no_{labels[i]}'], time, window)
        df.to_csv(out_path, index=False)
        print(f'Saved file excluding {labels[i]}')


def main_extract(filename: str, out_path: str, window: int):
    '''
    Extracts novelty, transience and resonance
    Writes to csv
    '''
    with open(filename) as f:
        emo_file = ndjson.load(f)
    
    data, time = get_data_time(emo_file)

    df = pd.DataFrame()
    df['date'] = time
    df['emo_prob'] = data
    df = extract_novelty_resonance(df, data, time, window)
    df.to_csv(out_path, index=False)


if __name__ == '__main__':
    # in_files = ['emo', 'pol']
    # out_files = ['polarity', 'emotion']
    # window = 7
    # for in_file, out_file in zip(in_files, out_files):
    #     filename = os.path.join("..", "summarized_emo", f"tweets_{in_file}_date.ndjson")
    #     out_path = os.path.join("..", "idmdl", f"tweets_{out_file}_date_W7.csv")
    #     main_extract(filename, out_path, window)

    filename = '../summarized_emo/tweets_recol_emo_date_hour_sd.ndjson'
    out_path = '../idmdl/tweets_recol_emotion_date_hour_12.csv'
    #Exlcude emotions
    # extract_excluded_emos(filename, out_path, window = 3)
    #Run regular
    main_extract(filename, out_path, window=12)
