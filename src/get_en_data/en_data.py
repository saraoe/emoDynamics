"""
Filtering combined and unzipped tweets by language
"""

import csv
import linecache
import time
import os
import argparse
from typing import Union
import pandas as pd

def filter_tweets(in_file:str, out_file:str, language:Union[str, list]=None, samples=None):
    """This function filters the unzipped tweets and saves in a new .tsv file

    Args:
        in_file (str): path to where the unzipped data is stored
        out_file (str): path to where the filtered data should be saved
        language (str or list, optional): string or list specifying language(s) to filter on. 
                                          Defaults to None, in which case no filtering is applied.
    """    
    filtered_tw = list()
    current_line = 1
    start = time.time()

    # Create list of single language or use input list
    if isinstance(language, str):
        lang_list=[language]
    else:
        lang_list=language
    langs = 0
    with open(in_file) as tsvfile:
        tsvreader = csv.reader(tsvfile, delimiter="\t")

        if current_line == 1: # Column names
            filtered_tw.append(linecache.getline(in_file, current_line))

            for line in tsvreader:
                if line[3] in lang_list: #if the tweet is of specified language, add it
                    filtered_tw.append(linecache.getline(in_file, current_line))
                    langs+=1
                current_line += 1
                current_time = time.time() - start
                if current_time >= 30:
                    print(f'processing line {current_line}')
                    start = time.time()

        print('\033[1mShowing first 5 tweets from the filtered dataset\033[0m')
        print(filtered_tw[1:(6 if len(filtered_tw) > 6 else len(filtered_tw))])
        print(f'Number of tweets with language classification as {language} is {langs}')

        # After cheking all tweets, add those of language specified to new file
        with open(out_file, 'w') as f_output:
            for item in filtered_tw:
                f_output.write(item)
    
    if samples:
        data = pd.read_csv(out_file, sep="\t")
        samp = data.sample(samples)
        samp.to_csv(f"{out_file.split('.')[0]}_samples.tsv", sep="\t")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--in_filepath', type=str, required=True,
                        help='Filepath for the input files.')
    parser.add_argument('--out_filepath', type=str, required=True,
                        help='Name of the output filepath.')
    parser.add_argument('--language', type=str, required=False, default=None,
                         help="Language of the tweets. Default is None, in which case no filtering happens.")
    args = parser.parse_args()

    in_file_path = args.in_filepath # 'clean_data_all_tweets.tsv'
    out_file_path = args.out_filepath
    language = args.language

    # df = pd.read_csv(in_file_path, sep="\t")
    # print(df.head(10))
    # language = df.lang.unique()
    # print(language)

    filter_tweets(in_file_path, out_file_path, language)


    # When authentication is set up using twitter_credentials.py, run 
    # python get_metadata.py -i clean_data_en_tweets.tsv -o hydrated_tweets -k api_keys.json