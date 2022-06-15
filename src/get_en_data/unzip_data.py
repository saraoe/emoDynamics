"""
Script for unzipping data from https://zenodo.org/record/6618186#.Yqc2SHZByvx
"""

import gzip
import shutil
import os

def unzip(in_file, out_file):
    """Unpacks the files from https://zenodo.org/record/6618186#.Yqc2SHZByvx

    Args:
        in_file (str): path to input file with combined zip files
        out_file (str): path to where output file should be stored
    """
    #Unzips the dataset and gets the TSV dataset
    with gzip.open(in_file, 'rb') as f_in:
        with open(out_file, 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)


# Download cleaned files and run in terminal: 
# can be downloaded with wget link_to_file
# cat full_dataset_clean_aa.tsv.gz full_dataset_clean_ab.tsv.gz full_dataset_clean_ac.tsv.gz full_dataset_clean_ad.tsv.gz > full_dataset_clean_all.tsv.gz

in_file = os.path.join("..", "..", "full_dataset_clean_all.tsv.gz")
out_file = 'clean_data_all_tweets.tsv'