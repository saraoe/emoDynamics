import pandas as pd
import csv
import linecache
import time
import os

df = pd.read_csv('clean_data_en_tweets.tsv',sep="\t")
print(df.head(10))
lang_list = df.lang.unique()


#Creates a new clean dataset with the specified language (pick from lang_list)
# filtered_language = "en"

# filtered_tw = list()
# current_line = 1
# start = time.time()
# with open("clean_data_all_tweets.tsv") as tsvfile:
#     tsvreader = csv.reader(tsvfile, delimiter="\t")

#     if current_line == 1:
#         filtered_tw.append(linecache.getline("clean_data_all_tweets.tsv", current_line))

#         for line in tsvreader:
#             if line[3] == filtered_language:
#                 filtered_tw.append(linecache.getline( "clean_data_all_tweets.tsv", current_line))
#             current_line += 1
#             current_time = time.time() - start
#             if current_time >= 30:
#                 print(f'processing line {current_line}')
#                 start = time.time()

#     print('\033[1mShowing first 5 tweets from the filtered dataset\033[0m')
#     print(filtered_tw[1:(6 if len(filtered_tw) > 6 else len(filtered_tw))])

#     with open('clean_data_en_tweets.tsv', 'w') as f_output:
#         for item in filtered_tw:
#             f_output.write(item)