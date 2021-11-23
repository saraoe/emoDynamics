'''
generator function for splitting text into max 512 tokens
'''

import spacy
import pandas as pd
import numpy as np
import time
from spacy_transformers.util import huggingface_tokenize
from transformers import AutoTokenizer


def texts_gen(df, text_column: str = "text", 
              tokenizer: str = "DaNLP/da-bert-tone-subjective-objective",
              model: str = "da_core_news_lg"):
    '''
    Text generator for input in spacy nlp pipeline
    Splits text so they don't exceed 500 tokens

    Args:
    - df: pandas dataframe with text
    - text_column: (str) name of column with text
    - tokenizer: (str) tokenizer of the model used
    - model: (str) spacy model

    yields id and text
    '''
    ids = list(range(0, len(df)))
    
    texts = df[text_column].tolist()
    tokenizer_ = AutoTokenizer.from_pretrained(tokenizer)
    nlp = spacy.load(model) 
    docs = nlp.pipe(texts) 

    #Now looping over docs
    for doc, id_ in zip(docs, ids):
        token_dict = huggingface_tokenize(tokenizer_, [doc.text])
        token = token_dict['input_ids'][0]
        if len(token) < 500:
            yield (doc.text ,id_)
        else: # if doc needs to be split
            bin_ = []
            sentence = []
            for sent in doc.sents:
                str_sent = str(sent)
                token_dict = huggingface_tokenize(tokenizer_, [str_sent])
                token = token_dict['input_ids'][0].tolist()
                if len(token) > 500:
                    print("This sentence is too long:", str_sent, sep = "\n")
                elif len(token) + len(bin_) > 500:
                    sentence = " ".join(sentence)
                    yield (sentence, id_)
                    bin_ = token
                    sentence = [str_sent]
                else:
                    bin_ += token
                    sentence.append(str_sent)
            sentence = " ".join(sentence)
            yield (sentence, id_)

