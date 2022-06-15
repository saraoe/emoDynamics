'''
Applying the methods of newsFluxus to BERT emotion distribution
- Novelty
- Transience
- Resonance 

Emotion distributions can be both eight emotions from BERT Emotion and polarity from BERT Tone
'''
import argparse
from typing import List

import ndjson, datetime
import pandas as pd

import os, sys
path = os.path.join("newsFluxus", "src")
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


def exclude_emotion_from_distribution(emo_distribution: List[int], index: int):
    '''
    Removes the emotion according to the index.
    Returns list with all the other emotions, with same relative size. 
    Sum of the returned list is still 1
    '''
    exclude_emo = emo_distribution[:index] + emo_distribution[index+1:]
    return [emo/sum(exclude_emo) for emo in exclude_emo]


def extract_excluded_emos(filename: str, 
                          out_folder: str, 
                          out_name: str, 
                          window: int, 
                          labels: List[str] = [
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
    for i, label in enumerate(labels):
        df = pd.DataFrame()
        df['date'] = time
        # df[f'no_{label}'] = [d[:i] + d[i+1:] for d in data]
        df[f'no_{label}'] = [exclude_emotion_from_distribution(d, i) for d in data]
        out_path = os.path.join(out_folder, f"{out_name}_W{window}_no_{label[:4]}.csv")
        df = extract_novelty_resonance(df, df[f'no_{label}'], time, window)
        df.to_csv(out_path, index=False)
        print(f'Saved file excluding {label}')


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


def main(filenames: List[str], window: int, extract_emos: str):
    if not os.path.exists("idmdl"):
        os.makedirs("idmdl")

    for file in filenames:
        filename = os.path.join("summarized_emo", f"{file}.ndjson")
        out_path = os.path.join("idmdl", f"{file}_W{window}.csv")
        main_extract(filename, out_path, window)
    
        if extract_emos == 'emo':
            out_folder = "idmdl"
            extract_excluded_emos(filename, out_folder, file, window)
        if extract_emos == 'pol':
            labels = ['positve', 'neutral', 'negative']
            out_folder = "idmdl"
            extract_excluded_emos(filename, out_folder, file, window, labels)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--filenames', type=str, required=True, nargs='+',
                        help='Filenames of the input files')
    parser.add_argument('--window', type=int, required=False, default=3,
                        help='Size of the window')
    parser.add_argument('--extract_emotions', type=str, required=False, default=None,
                        help='''If defined the emotion dynamics signal for the individual emotions are calculated. 
                                The argument must be either "emo" or "pol", detmining whether the emotions come 
                                from BERT emotion or BERT Tone.''')
    args = parser.parse_args()

    print(f'''Running emotionFluxus.py with:
             filenames={args.filenames},
             window={args.window},
             extract_emos={args.extract_emotions}''')
    main(filenames=args.filenames,
         window=args.window,
         extract_emos=args.extract_emotions)
