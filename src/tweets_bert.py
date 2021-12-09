'''
Runing Bert model on tweets

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
import preprocess
import spacy
import dacy
from spacy.tokens import Doc
from dacy.sentiment import da_vader_getter, add_berttone_subjectivity, add_bertemotion_emo, add_bertemotion_laden, add_berttone_polarity
import time
from csv import writer
spacy.require_gpu()

## define functions ## 
def append_list_as_row(file_name, list_of_elem):
    # Open file in append mode
    with open(file_name, 'a+', newline='') as write_obj:
        # Create a writer object from csv module
        csv_writer = writer(write_obj)
        # Add contents of list as last row in the csv file
        csv_writer.writerow(list_of_elem)


def ndjson_gen(filepath: str):
    for in_file in glob(filepath):
        with open(in_file) as f:
            reader = ndjson.reader(f)

            for post in reader:
                yield post


def gen_to_tuple_gen(generator,field="text"):
    for post in generator:  
        if "^RT" in post[field]: # remove retweets
            continue
        post[field] = preprocess.clean_tweets(post[field])
        yield post[field], post


def main(in_filepath: str, out_file: str):
    ## All models ##
    print('Prepare models')
    nlp = spacy.load("da_core_news_lg") 
    Doc.set_extension("vader_da", getter=da_vader_getter, force = True)
    nlp = add_berttone_subjectivity(nlp)
    nlp = add_bertemotion_laden(nlp)   
    nlp = add_bertemotion_emo(nlp) 
    nlp = add_berttone_polarity(nlp)#, force_extension=True) 

    ### Creating out csv ###
    out = pd.DataFrame(columns = ["created_at", "id",
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
    out.to_csv(f'/home/commando/stine-sara/data/{out_file}.csv')

    gen = ndjson_gen(in_filepath)
    tuple_gen = gen_to_tuple_gen(gen)
    docs = nlp.pipe(tuple_gen, as_tuples=True, batch_size = 1024)

    model_time = time.time()
    for index, (doc, context) in enumerate(docs):
        if not doc: # if doc is an empty string
            continue
        
        #Sentiment
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
        row = [index,
            context["created_at"],
            context["id"],
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

        append_list_as_row(f'/home/commando/stine-sara/data/{out_file}.csv', row)
        mid_time = time.time()
        if index % 100 == 0: 
            print(f'Running model on row number {index} now finished - time in min: {(mid_time - model_time)/60}')



if __name__=='__main__':
    print("Starting time")
    time_start = time.time()

    in_filepath = '/data/004_twitter-stopword/*.ndjson'
    out_file = 'emotion_tweets_2021'
    main(in_filepath, out_file)

    time_end = time.time() 
    total_time = time_end - time_start
    print(f'total time spent is {total_time/60} minutes')
