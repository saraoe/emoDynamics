'''
Runing Bert model on News papers
- Frontpages only

Models:
- Vader
- Bert subjectivity
- Bert emotional laden
- Bert emotion
- Bert polarity
'''
### Load modules ###
print("Importing modules")
import ndjson
from glob import glob
import pandas as pd
import numpy as np
import spacy
import dacy
from spacy.tokens import Doc
from dacy.sentiment import da_vader_getter, add_berttone_subjectivity, add_bertemotion_emo, add_bertemotion_laden, add_berttone_polarity
import time
from csv import writer 
from tensor_fix import texts_gen

from tqdm import tqdm
from wasabi import msg

spacy.require_gpu()
print("Starting time")
time_start = time.time()

## SETUP ##
filename = 'emotion_news_fp'

# define functions 
def append_list_as_row(file_name, list_of_elem):
    # Open file in append mode
    with open(file_name, 'a+', newline='') as write_obj:
        # Create a writer object from csv module
        csv_writer = writer(write_obj)
        # Add contents of list as last row in the csv file
        csv_writer.writerow(list_of_elem)

print('Reading in files')
df = pd.DataFrame(columns = ["date", "title", "text", "paper"])
for file_name in glob('/home/commando/stine-sara/data/210906_hope_frontpages/preprocessed/*.ndjson'):
    with open(file_name) as f:
        data = ndjson.load(f)
        frame = pd.DataFrame(data)
        frame["paper"] = str(file_name[-10:-7]) #Also add column with which paper
        df = df.append(frame)
df = df.set_index(np.arange(len(df)))

### Creating out csv ###
out = pd.DataFrame(columns = ["paper", "date",
                              "Sentiment_compound",
                              "Sentiment_neutral",
                              "Sentiment_negative",
                              "Sentiment_positive",
                              "Bert_subj_label",
                              "Bert_subj_prob",
                              "Bert_emo_laden",
                              "Bert_emo_laden_prob",
                              "Bert_emo_emotion",
                              "Bert_emo_emotion_prob", 
                              "polarity", 
                              "polarity_prob"])
out.to_csv(f'{filename}.csv')

## All models ##
print('Prepare models')
nlp = spacy.load("da_core_news_lg") 
Doc.set_extension("vader_da", getter=da_vader_getter, force = True)
nlp = add_berttone_subjectivity(nlp)
nlp = add_bertemotion_laden(nlp)   
nlp = add_bertemotion_emo(nlp) 
nlp = add_berttone_polarity(nlp)#, force_extension=True) 

# docs = nlp.pipe(df["text"], batch_size = 1024)
docs = nlp.pipe(texts_gen(df), as_tuples = True)
model_time = time.time()

for doc, index in docs:
    if not doc:
        continue
    #Sentiment
    sents = doc.sents
    d = doc._.vader_da
    
    #Bert subjectivity
    subj_label = doc._.subjectivity 
    subj_prob = doc._.subjectivity_prop["prop"]
    
    #Bert emotion
    laden = doc._.laden
    emo = doc._.emotion
    laden_prob = doc._.laden_prop
    emo_prob = doc._.emotion_prop
    
    #polarity 
    pol_label = doc._.polarity
    pol_label_prob = doc._.polarity_prop["prop"]
    
    #creating row
    assert isinstance(df["paper"][index], str), "Paper of this row is not a string {index}"
    row = [index,
        df["paper"][index],
        df["date"][index],
        d["compound"],
        d["neu"],
        d["neg"],
        d["pos"],
        subj_label,
        max(subj_prob),
        laden,
        max(laden_prob["prop"]),
        emo,
        emo_prob["prop"],
        pol_label,
        pol_label_prob]

    append_list_as_row(f'{filename}.csv', row)
    mid_time = time.time()
    if index % 100 == 0: 
        print(f'Running model on row number {index} now finished - time in min: {(mid_time - model_time)/60}')

time_end = time.time() 
total_time = time_end - time_start
print(f'total time spent is {total_time/60} minutes')