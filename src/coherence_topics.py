"""
Estimation the optimal number of topics for the topic modelling using gensims coherence model
"""
from glob import glob
import ndjson, re, random
import numpy as np
import pandas as pd
from tweetopic import DMM, TopicPipeline
from sklearn.feature_extraction.text import CountVectorizer
from tmtoolkit.topicmod.evaluate import metric_coherence_gensim
import matplotlib.pyplot as plt
import time
import functools, multiprocessing


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


def random_sample(x, x_len, size):
    """
    Takes in an iterable x and returns a generator of size with random element from x

    Args:
        x (Iterable): iterable from with the elements to fill the list
        x_len (int): length of x
        size (int): size of the output list

    Returns
        generator
    """
    indexes = sorted(random.sample([i for i in range(x_len)], size))
    # indexes = [i for i in range(size)]
    time_ = time.time()
    for i, elem in enumerate(x):
        if i in indexes:
            indexes.remove(i)
            yield elem
            if not indexes:
                break
        if i % 100000 == 0:
            print(f"at tweet {i}, havent yielded {len(indexes)} tweets yet")
            print(f"time = {time.time()-time_}")
            time_ = time.time()


def tokenize(doc, vocab, regex):
    tokens = re.findall(regex, doc.lower())
    return [token for token in tokens if token in vocab]


def main(path, n_texts, measure, language):
    print("getting sample")
    # sample 100,000 tweets
    tweets = [
        tweet for tweet in random_sample(text_gen(path), x_len=n_texts, size=100000)
    ]

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

    print("vectorizing and tokenizing tweets")
    # get arguments for metric_gensim_coherence
    vectorizer.fit(tweets)
    n = len(vectorizer.vocabulary_)
    vec_transform = vectorizer.transform(tweets)
    vocab = np.array([x for x in vectorizer.vocabulary_.keys()])
    regex = re.compile("(?u)\\b\\w\\w+\\b")
    pool = multiprocessing.Pool(5)
    tok_tweets = pool.map(functools.partial(tokenize, vocab=vocab, regex=regex), tweets)
    # tok_tweets = tokenizer(tweets, vocab)

    n_topics = [n * 10 for n in range(1, 30)]
    coherences = []

    for i, n_topic in enumerate(n_topics, start=1):
        print(f"fitting model with {n_topic} topics")
        dmm = DMM(
            n_components=n_topic,
            n_iterations=200,
            alpha=0.1,
            beta=0.1,
        )

        pipeline = TopicPipeline(vectorizer, dmm)
        pipeline.fit(tweets)
        top_words = pipeline.top_words(n)

        cm = metric_coherence_gensim(
            measure=measure,
            top_n=10,
            topic_word_distrib=pd.DataFrame(top_words).values,
            dtm=vec_transform,
            vocab=vocab,
            texts=tok_tweets,
            return_coh_model=True,
        )
        coherences.append(cm.get_coherence())

        # plot result
        plt.plot(n_topics[:i], coherences, "o", color="black")
        plt.plot(n_topics[:i], coherences, "--", color="grey")
        plt.xlabel("Number of topics")
        plt.ylabel(f"Coherence ({measure})")
        plt.savefig(f"fig/topic_coherence_{measure}.png")


if __name__ == "__main__":
    path = "/data/004_twitter-stopword/*.ndjson"
    n_texts = 43555075
    language = "da"
    measure = "c_v"

    print(
        f"""Running coherence_topics.py with:
             path={path},
             n_texts={n_texts},
             measure={measure},
             language={language}"""
    )

    main(path, n_texts, measure, language)
