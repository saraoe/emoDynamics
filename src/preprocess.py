'''
Functions for preprocessing the tweets

These are modified versions of functions written by Maris Sala
https://github.com/marissala/HOPE-keyword-query-Twitter
'''
### Load modules ###
import re

### Functions ###
def remove_retweets(data):
    """Finds tweets that are RTs and removes them
    data: pandas DataFrame with (at leat) column "text"
    From Maris code: extract_data.py
    """
    patternDel = "^RT"
    data["text"] = data["text"].astype(str)
    filtering = data['text'].str.contains(patternDel)
    removed_RT = data[~filtering].reset_index(drop=True)
    return removed_RT


def remove_emoji(string:str):
    """Remove all emojis (captures a lot but not everything)
    string: str
    From Maris code: preprocess_stats.py
    """
    emoji_pattern = re.compile("["
                               u"\U0001F600-\U0001F64F"  # emoticons
                               u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                               u"\U0001F680-\U0001F6FF"  # transport & map symbols
                               u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                               u"\U00002500-\U00002BEF"  # chinese char
                               u"\U00002702-\U000027B0"
                               u"\U00002702-\U000027B0"
                               u"\U000024C2-\U0001F251"
                               u"\U0001f926-\U0001f937"
                               u"\U00010000-\U0010ffff"
                               u"\u2640-\u2642"
                               u"\u2600-\u2B55"
                               u"\u200d"
                               u"\u23cf"
                               u"\u23e9"
                               u"\u231a"
                               u"\ufe0f"  # dingbats
                               u"\u3030"
                               "]+", flags=re.UNICODE)
    return emoji_pattern.sub(r'', string)


def clean_tweets(tweet):
    """Remove mentions, hashtags, URLs, emojis
    row: pandas DataFrame row with column "text"
    From Maris code: preprocess_stats.py
    """
    clean_tweet = re.sub(r'@(\S*)\w', '', tweet) #mentions
    clean_tweet = re.sub(r'#\S*\w', '', clean_tweet) # hashtags
    # Remove URLs
    url_pattern = re.compile(
        r'(https?:\/\/(?:www\.|(?!www))[a-zA-Z0-9][a-zA-Z0-9-]+[a-zA-Z0-9]\.[^\s]{2,}|www\.[a-zA-Z0-9][a-zA-Z0-9-]+[a-zA-Z0-9]\.[^\s]{2,}|https?:\/\/(?:www\.|(?!www))[a-zA-Z0-9]+\.[^\s]{2,}|www\.[a-zA-Z0-9]+\.[^\s]{2,})')
    clean_tweet = re.sub(url_pattern, '', clean_tweet)
    clean_tweet = remove_emoji(clean_tweet)
    return clean_tweet


def remove_quote_tweets(df):
    """Creates mentioneless_text, remove bot-like tweets/quote tweets, when 50 first characters are the exact
    same, remove as duplicates
    df: pandas DataFrame with column "text"
    From Maris code: preprocess_stats.py
    """
    df["text"] = df["text"].astype(str)
    df["mentioneless_text"] = df.apply(lambda row: remove_mentions(row), axis = 1)
    # print("Generated mentioneless texts")
    df["text50"] = df["mentioneless_text"].str[0:50]
    
    df["dupe50"] = df["text50"].duplicated(keep = "first")

    # print("Length of quote tweets: ")
    
    df = df[df["dupe50"] == False].reset_index()
    return df
