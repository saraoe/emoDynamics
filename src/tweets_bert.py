"""
Bert emotion classification on tweets

Models for danish tweets:
- Vader
- Bert subjectivity
- Bert emotional laden
- Bert emotion
- Bert polarity

Model for english tweets:
- distilbert-base-uncased-emotion
"""

### Load modules ###
import argparse
import ndjson
from glob import glob
import re
import pandas as pd
import preprocess
import spacy
from spacy.tokens import Doc
import time
from csv import writer

spacy.require_gpu()

## define functions ##
def append_list_as_row(file_name, list_of_elem):
    # Open file in append mode
    with open(file_name, "a+", newline="") as write_obj:
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


def gen_to_tuple_gen(generator, field="text"):
    for post in generator:
        if not post:  # remove empty posts
            continue
        if re.search("^RT", post[field]):  # remove retweets
            continue
        post[field] = preprocess.clean_tweets(post[field])
        yield post[field], post


def main(in_filepath: str, out_filepath: str, language: str):
    ## All models ##
    print("Prepare models")
    if language == "da":
        import ssl

        ssl._create_default_https_context = ssl._create_unverified_context

        from dacy.sentiment import (
            da_vader_getter,
            add_berttone_subjectivity,
            add_bertemotion_emo,
            add_bertemotion_laden,
            add_berttone_polarity,
        )

        # nlp = spacy.load("da_core_news_lg")
        nlp = spacy.blank('da')
        Doc.set_extension("vader_da", getter=da_vader_getter, force=True)
        nlp = add_berttone_subjectivity(nlp)
        nlp = add_bertemotion_laden(nlp)
        nlp = add_bertemotion_emo(nlp)
        nlp = add_berttone_polarity(nlp)  # , force_extension=True)
    if language == "en":
        import spacy_wrap

        nlp = spacy.blank("en")
        config = {
            "doc_extension_trf_data": "clf_trf_data",
            "doc_extension_prediction": "emotion",
            "labels": ["sadness", "joy", "love", "anger", "fear", "surprise"],
            "model": {
                "name": "bhadresh-savani/distilbert-base-uncased-emotion",  # the model name or path of huggingface model
            },
        }
        nlp.add_pipe("classification_transformer", config=config)

    ### Creating out csv ###
    if language == "da":
        out = pd.DataFrame(
            columns=[
                "created_at",
                "id",
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
                "polarity_prob",
            ]
        )
    if language == "en":
        out = pd.DataFrame(
            columns=["created_at", "id", "emotion_label", "emotion_prob"]
        )
    out.to_csv(f"{out_filepath}.csv")

    gen = ndjson_gen(in_filepath)
    tuple_gen = gen_to_tuple_gen(gen)
    docs = nlp.pipe(tuple_gen, as_tuples=True, batch_size=1024)

    model_time = time.time()
    for index, (doc, context) in enumerate(docs):
        if not doc:  # if doc is an empty string
            continue

        if language == "da":
            # Sentiment
            d = doc._.vader_da

            # Bert subjectivity
            subj_label = doc._.subjectivity
            subj_prob = doc._.subjectivity_prop["prop"]

            # Bert emotion
            laden = doc._.laden
            emo = doc._.emotion
            laden_prob = doc._.laden_prop
            emo_prob = doc._.emotion_prop

            # polarity
            pol_label = doc._.polarity
            pol_label_prob = doc._.polarity_prop["prop"]

            # creating row
            row = [
                index,
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
                pol_label_prob,
            ]

        if language == "en":
            # emotion
            emo = doc._.emotion
            emo_prob = doc._.emotion_prob["prob"]

            # creating row
            row = [index, context["created_at"], context["id"], emo, emo_prob]

        append_list_as_row(f"{out_filepath}.csv", row)
        mid_time = time.time()
        if index % 10000 == 0:
            print(
                f"Running model on row number {index} now finished - time in min: {(mid_time - model_time)/60}"
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--in_filepath", type=str, required=True, help="Filepath for the input files."
    )
    parser.add_argument(
        "--out_filepath", type=str, required=True, help="Name of the output filepath."
    )
    parser.add_argument(
        "--language",
        type=str,
        required=False,
        default="da",
        help="Language of the tweets. Either da or en. Default is da.",
    )
    args = parser.parse_args()

    print("Starting time")
    time_start = time.time()

    in_filepath = args.in_filepath + "*.ndjson"
    out_filepath = args.out_filepath
    language = args.language

    print(
        f"""Running tweets_bert.py with:
             in_filepath = {in_filepath},
             out_filepath= {out_filepath},
             language= {language}"""
    )
    main(in_filepath, out_filepath, language)

    time_end = time.time()
    total_time = time_end - time_start
    print(f"total time spent is {total_time/60} minutes")
