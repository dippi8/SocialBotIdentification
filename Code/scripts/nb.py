import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
import re
from sklearn.pipeline import Pipeline
import nltk
from nltk.stem.snowball import SnowballStemmer
from sklearn.metrics import f1_score
from nltk.corpus import stopwords
import pickle
import tweepy
import sys

user = sys.argv[1]


# create connection with Twitter API

CONSUMER_KEY = 'J3uhwWHdmSt3uD69ry8r2kc3B'
CONSUMER_SECRET = 'LeuM7dVfFbXH0bc5fFkZIxlNiUXyjIx4Kcjh58HeWs2TrLijDx'
ACCESS_TOKEN = '327497511-5dNjiGaTQHASljoxNr1qlMOlHrrSB21HBFBDjx2E'
ACCESS_TOKEN_SECRET = 'tHLqwlhHYJWptW3femyq0rbMW6ZItu5yLQ3DFGJqJ8Xlg'

auth = tweepy.OAuthHandler(CONSUMER_KEY, CONSUMER_SECRET)
auth.set_access_token(ACCESS_TOKEN, ACCESS_TOKEN_SECRET)
api = tweepy.API(auth)


# download user's tweets

stuff = api.user_timeline(user_id = user, count = 100, include_rts = True, tweet_mode="extended")
tweets = []

for tweet in stuff:
    tweets.append(tweet._json['full_text'])
    
tweets = pd.Series(tweets)


# initialize pipeline

stemmer = SnowballStemmer("english", ignore_stopwords=True)

class StemmedCountVectorizer(CountVectorizer):
    def build_analyzer(self):
        analyzer = super(CountVectorizer, self).build_analyzer()
        return lambda doc:(stemmer.stem(w) for w in analyzer(doc))



# load model

nb = pickle.load( open( "nb.model", "rb" ) )


# define preprocess functions

def remove_rt(x):
    
    if 'RT' in x:
        x = x.replace('RT', '')
        try:
            return x[x.rindex(':')+2:]
        except:
            return x
    else:
        return x


stop_words = stopwords.words('english')

def remove_stop(x):
    return [word for word in x.split() if word not in stop_words]


# preprocess tweets

tweets = tweets.apply(lambda x: remove_rt(x))
tweets = tweets.apply(lambda x: re.sub(r'^\/t.co\/[^\s]+', '', x))
tweets = tweets.apply(lambda x: re.sub(r'[^\w\s]','',x))
tweets = tweets.apply(lambda x: x.lower())
tweets = tweets.apply(lambda x: remove_stop(x))
tweets = tweets.astype(str)
tweets = tweets[tweets!='[]']


# perform predictions over tweets

pred = nb.predict_proba(tweets)


# return average of predictions

print(np.mean(pred, axis=0).tolist())