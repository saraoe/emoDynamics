"""
Topic modelling on tweets using tweetopic
"""
from glob import glob
import ndjson, re
from csv import writer
from tweetopic import DMM, TopicPipeline
from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd
import time
import argparse


def ndjson_gen(filepath: str, text_field: str = "text"):
    for in_file in glob(filepath):
        with open(in_file) as f:
            reader = ndjson.reader(f)

            for post in reader:
                if not post:  # remove empty posts
                    continue
                if re.search("^RT", post[text_field]):  # remove retweets
                    continue
                yield post


def text_gen(filepath: str, text_field: str = "text"):
    for post in ndjson_gen(filepath):
        yield post[text_field]


def append_list_as_row(file_name, list_of_elem):
    # Open file in append mode
    with open(file_name, "a+", newline="") as write_obj:
        # Create a writer object from csv module
        csv_writer = writer(write_obj)
        # Add contents of list as last row in the csv file
        csv_writer.writerow(list_of_elem)


def main(in_filepath: str, out_filepath: str, n_topics: int, language: str):
    if language == "da":
        file = open("src/stop_words.txt", "r+")
        stop_words = file.read().split()
    if language == "en":
        stop_words = "english"

    vectorizer = CountVectorizer(
        stop_words=stop_words,
        max_df=0.5,
        min_df=1,
    )
    dmm = DMM(
        n_components=n_topics,
        n_iterations=200,
        alpha=0.1,
        beta=0.1,
    )

    print("------- \nfitting topics \n-------")
    start_time = time.time()
    pipeline = TopicPipeline(vectorizer, dmm)
    tweets = text_gen(in_filepath)
    topics = pipeline.fit_transform(tweets)
    print(f"Number of tweets = {len(topics)}")
    print(f"Top 3 words in topics: \n {pipeline.top_words(top_n=3)}")
    print(f"time fitting model in min: {(time.time() - start_time)/60}")

    # write csv
    print("------- \nstart writing csv \n-------")
    start_time = time.time()
    out = pd.DataFrame(columns=["created_at", "id", "topic_prob"])
    out.to_csv(f"{out_filepath}.csv")
    for index, (post, topic) in enumerate(zip(ndjson_gen(in_filepath), topics)):
        row = [index, post["created_at"], post["id"], topic]
        append_list_as_row(f"{out_filepath}.csv", row)
        mid_time = time.time()
        if index % 10000 == 0:
            print(
                f"row number {index} now finished - time in min: {(mid_time - start_time)/60}"
            )
    print("Done with all tweets")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--in_filepath", type=str, required=True, help="Filepath for the input files."
    )
    parser.add_argument(
        "--out_filepath", type=str, required=True, help="Name of the output filepath."
    )
    parser.add_argument(
        "--n_topics",
        type=int,
        required=True,
        help="Number of topics to fit.",
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
    n_topics = args.n_topics
    language = args.language

    print(
        f"""Running tweets_topic.py with:
             in_filepath = {in_filepath},
             out_filepath= {out_filepath},
             n_topics= {n_topics},
             language= {language}"""
    )

    main(in_filepath, out_filepath, n_topics, language)
