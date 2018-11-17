import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
import re
from sklearn.pipeline import Pipeline
import nltk
from nltk.stem.snowball import SnowballStemmer
from sklearn.metrics import f1_score
from nltk.corpus import stopwords
import pickle
import tweepy
import sys
from datetime import datetime
import os
import tensorflow as tf
import urllib.request
import urllib.parse as urlparse
from ast import literal_eval

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

stuff = api.user_timeline(screen_name = user, count = 100, include_rts = True, tweet_mode="extended")
tweets = []

for tweet in stuff:
    tweet._json['user_id'] = tweet._json['user']['id']
    if len(tweet._json['entities']['urls']) != 0:
        tweet._json['url'] = tweet._json['entities']['urls'][0]['expanded_url']
    else:
         tweet._json['url'] = None
    del tweet._json['user'], tweet._json['entities']

    tweets.append(tweet._json)
        
tweets = pd.DataFrame.from_dict(tweets)


# download user's features

user = api.get_user(user)

user_features = pd.DataFrame([[user.id,
                    user.name,
                    user.screen_name,
                    user.statuses_count,
                    user.followers_count,
                    user.friends_count,
                    user.favourites_count,
                    user.listed_count,
                    user.url,
                    user.lang,
                    user.time_zone,
                    user.location,
                    user.default_profile,
                    user.default_profile_image,
                    user.geo_enabled,
                    user.profile_image_url,
                    user.profile_use_background_image,
                    user.profile_background_image_url_https,
                    user.profile_text_color,
                    user.profile_image_url_https,
                    user.profile_sidebar_border_color,
                    user.profile_background_tile,
                    user.profile_sidebar_fill_color,
                    user.profile_background_image_url,
                    user.profile_background_color,
                    user.profile_link_color,
                    user.utc_offset,
                    user.is_translator,
                    user.follow_request_sent,
                    user.protected,
                    user.verified,
                    user.notifications,
                    user.description,
                    user.contributors_enabled,
                    user.following,
                    user.created_at]],
                    columns=["id","name","screen_name","statuses_count","followers_count","friends_count","favourites_count","listed_count","url","lang","time_zone","location","default_profile","default_profile_image","geo_enabled","profile_image_url","profile_use_background_image","profile_background_image_url_https","profile_text_color","profile_image_url_https","profile_sidebar_border_color","profile_background_tile","profile_sidebar_fill_color","profile_background_image_url","profile_background_color","profile_link_color","utc_offset","is_translator","follow_request_sent","protected","verified","notifications","description","contributors_enabled","following","created_at"]
                  )


# compute context score

porn_words = pd.read_csv('data/nsfw/filtered_main_words.csv', sep=',')
prop_words = pd.read_csv('data/propaganda/filtered_main_words.csv', sep=',')
spam_words = pd.read_csv('data/spam/filtered_main_words.csv', sep=',')
fake_words = pd.read_csv('data/fake_followers/filtered_main_words.csv', sep=',')

bots_words = pd.read_csv('data/bots/filtered_main_words.csv', sep=',')
gen_words = pd.read_csv('data/genuine/filtered_main_words.csv', sep=',')

def compute_score(tweets):

    user_score = pd.DataFrame(columns=['NSFW_words_score','news_spreaders_words_score','spam_bots_words_score','fake_followers_words_score','bot_words_score','gen_words_score'])

    for tweet in tweets['full_text']:
        # check for words in main_words and compute the scores for each tweet and for each category
        mask = np.in1d(porn_words.word, tweet.split())
        porn_score = porn_words.loc[mask]['score'].values.sum()
        mask = np.in1d(prop_words.word, tweet.split())
        prop_score = prop_words.loc[mask]['score'].values.sum()
        mask = np.in1d(spam_words.word, tweet.split())
        spam_score = spam_words.loc[mask]['score'].values.sum()
        mask = np.in1d(fake_words.word, tweet.split())
        fake_score = fake_words.loc[mask]['score'].values.sum()
        mask = np.in1d(gen_words.word, tweet.split())
        genuine_score = gen_words.loc[mask]['score'].values.sum()
        mask = np.in1d(bots_words.word, tweet.split())
        bots_score = bots_words.loc[mask]['score'].values.sum()
        
        
        user_score = user_score.append(pd.DataFrame({'NSFW_words_score': porn_score, 'news_spreaders_words_score': prop_score, 'spam_bots_words_score': spam_score,'fake_followers_words_score': fake_score, 'bot_words_score':bots_score, 'gen_words_score':genuine_score}, index=[0]), sort=False, ignore_index=True)

    return user_score

if len(tweets) > 0:
    # sum all the scores of each category
    user_score = compute_score(tweets).sum()
    scores = np.divide(user_score,len(tweets))
else:
    scores = pd.DataFrame({'NSFW_words_score': 0, 'news_spreaders_words_score': 0, 'spam_bots_words_score': 0, 'fake_followers_words_score': 0, 'bot_words_score':0, 'gen_words_score':0}, index=[0]).T

scores = pd.DataFrame(scores).T

# compute intradistances

def compute_centroid(tf_idf):

    center = tf_idf.sum(axis=1)/tf_idf.shape[0]
    return center

def dist_from_centroid(tf_idf, centroid):
    
    distances = []
    for elem in tf_idf:
        distances.append(np.linalg.norm(tf_idf - centroid))
    return distances

def wss(id, tweets_df, is_tweet = 1):
    
    if is_tweet == 1:
        # get tweets per id
        vector = tweets_df['full_text']
        n_vectors = len(vector)
    elif is_tweet == 0:
        # get domains per id
        vector = tweets_df['url']
        vector = vector.fillna('').astype(str)
        for i in range(len(vector)):
            vector.iloc[i] = urlparse(vector.iloc[i]).netloc
        n_vectors = len(vector)
    else:
        print ('Invalid Input')

    transformer = TfidfVectorizer(smooth_idf=True)
    tf_idf = transformer.fit_transform(vector).todense()
    
    centroid = compute_centroid(tf_idf)
    distances = dist_from_centroid(tf_idf, centroid)
    avg_dist = np.asarray(distances).sum()/n_vectors
    
    return avg_dist

def intradistances():
    try:
        tw = (wss(user, tweets, 1))
    except:
        tw = 0
        
    return  tw

intradistance = intradistances()


def get_url(df):
    
    expanded_urls = []
    for x in df.itertuples():
        try:
            if len(x.url)>0:
                expanded_urls.append(urlparse.urlparse(x.url).netloc)

        except:
            pass
            
        expanded_urls = list(map(lambda x: x.replace('www.','').replace('/',''), expanded_urls))
    return expanded_urls

import collections

def compute_entropy():

    #urls = tweets['url']
    #is_quote = tweets['is_quote_status']
    if len(tweets) > 0:
	    param = tweets[['url','is_quote_status']]
	    
	    if len(param) > 0:
	        urls = get_url(param)
	        
	        # remove empty strings
	        urls = [x for x in urls if x]
	    
	        if len(urls) > 0:
	            # count frequency
	            counter=collections.Counter(urls)
	            occurrences = np.array(list(counter.values()))

	            # update N
	            N = occurrences.sum()

	            p = occurrences/N
	            entropies = -p *np.log2(p)
	            entropy = entropies.sum()
	        else:
	            entropy = 0.0
	    else:
	        entropy = 0.0

    else:
    	entropy = 0.0
        
    return float(entropy)


url_entropy = compute_entropy()

intradistances = pd.DataFrame({'url_entropy': url_entropy, 'tweet_intradistance': intradistance},index=[0])

# compute unreliability rate

domains = pickle.load(open('data/fake_url.sources','rb'))
urls = get_url(tweets)
unreliability_rate = np.in1d(urls,domains).astype(int).sum() / len(urls)

# NSFW features

def create_labels():
	# Loads label file, strips off carriage return
	label_lines = [line.rstrip() for line 
	                   in tf.gfile.GFile("models/NSFW-detection/retrained_labels.txt")]


def create_graph():
    # Unpersists graph from file
    with tf.gfile.FastGFile("models/NSFW-detection/retrained_graph.pb", 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        tf.import_graph_def(graph_def, name='')


def nsfw(url, sess):
    try:
        urllib.request.urlretrieve(url, "local-filename.jpg")
        image_path = 'local-filename.jpg'
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

        # Read in the image_data
        image_data = tf.gfile.FastGFile(image_path, 'rb').read()

        
        # Feed the image_data as input to the graph and get first prediction
        softmax_tensor = sess.graph.get_tensor_by_name('final_result:0')
        predictions = sess.run(softmax_tensor, {'DecodeJpeg/contents:0': image_data})

        return predictions[0][1]
    except:
        return 0


def nsfw_detection(bot_id):
 
    with tf.Session() as sess:
        porn, tot = 0, 0
        try:
            for media in tweets.extended_entities[tweets.extended_entities.notnull()][:10]:

                try:
                    url = media['media'][0]['media_url_https']
                    porno_score = nsfw(url,sess)
                    if porno_score > 0.8:
                        porn += 1
                    tot+=1

                    #print(url, porno_score)

                except:
                    print('exception')

            if tot > 0:
                nudity = porn/tot
            else:
                nudity = 0
        except:
            nudity = 0
        try:
            profile = user.profile_image_url_https.replace('normal', '400x400')
            profile_nudity = nsfw(profile, sess)
        except:
            profile_nudity = 0

        return profile_nudity, nudity


create_labels()
create_graph()

nsfw_profile, nsfw_avg = nsfw_detection(user.id)

nsfw_data = pd.DataFrame({'nsfw_profile': nsfw_profile, 'nsfw_avg': nsfw_avg},index=[0])

os.unlink('local-filename.jpg')

# collect descriptive features

def describe_tweets(tweets):
    
    ret_perc, media_perc, url_perc, quote_perc = tweet_perc(tweets)
    
    avg_len, avg_ret, avg_fav, avg_hash = tweet_desc(tweets, 'avg')
    max_len, max_ret, max_fav, max_hash = tweet_desc(tweets, 'max')
    min_len, min_ret, min_fav, min_hash = tweet_desc(tweets, 'min')
    
    freq = tweet_freq(tweets)

    frame = np.array([avg_len, max_len, min_len, avg_ret, max_ret, min_ret, avg_fav, max_fav, min_fav, avg_hash, max_hash, min_hash, freq, ret_perc, media_perc, url_perc, quote_perc])
   
    desc_features = pd.DataFrame({'avg_len': avg_len, 'max_len': max_len, 'min_len': min_len, 'avg_ret': avg_ret, 'max_ret': max_ret, 'min_ret': min_ret, 'avg_fav': avg_fav, 'max_fav': max_fav, 'min_fav': min_fav, 'avg_hash' : avg_hash, 'max_hash' : max_hash, 'min_hash' : max_hash,'freq': freq, 'ret_perc': ret_perc, 'media_perc': media_perc, 'url_perc': url_perc, 'quote_perc': quote_perc}, index=[0])
    
    return desc_features

def tweet_perc(tweets):
    
    try:
        ret_perc = np.invert(tweets.retweeted_status.isnull()).sum()/len(tweets)
    except:
        ret_perc = 0
    try:
        media_perc = np.invert(tweets.extended_entities.isnull()).sum()/len(tweets)
    except:
        media_perc = 0
    try:
        url_perc = np.invert(tweets.url.isnull()).sum()/len(tweets)
    except:
        url_perc = 0
    try:
        quote_perc = tweets.is_quote_status.sum()/len(tweets)
    except:
        quote_perc = 0
    
    return ret_perc, media_perc, url_perc, quote_perc

def hashtag_count(tweets):
    
    occurrences = []
    for tweet in tweets:
        occurrences.append(tweet.count('#'))
        
    return occurrences

def tweet_desc(tweets, metric):
    
    tweets_lenght = tweets['full_text'].apply(lambda x: len(x))
    
    if metric == 'avg':
        ret = np.mean(tweets.retweet_count)
        lenght = np.mean(tweets_lenght)
        fav = np.mean(tweets.favorite_count)
        hashtag = np.mean(hashtag_count(tweets['full_text']))
    elif metric == 'max':
        ret = max(tweets.retweet_count)
        lenght = max(tweets_lenght)
        fav = max(tweets.favorite_count)
        hashtag = max(hashtag_count(tweets['full_text']))
    elif metric == 'min':
        ret = min(tweets.retweet_count)
        lenght = min(tweets_lenght)
        fav = min(tweets.favorite_count)
        hashtag = min(hashtag_count(tweets['full_text']))

    return lenght, ret, fav, hashtag

def tweet_freq(tweets):
    
    dates = list(tweets.created_at)
    
    last = dates[0]
    d = last[8:10]
    m = last[4:7]
    y = last[-4:]
    date = d + ' ' + m + ' ' + y
    last = datetime.strptime(date, '%d %b %Y')
    
    first = dates[-1]
    d = first[8:10]
    m = first[4:7]
    y = first[-4:]
    date = d + ' ' + m + ' ' + y
    first = datetime.strptime(date, '%d %b %Y')
    
    delta = (last - first).days + 1
    freq = len(tweets)/delta
    
    return freq

def describe(tweets_df):
    
    tweets = tweets_df
    
    if len(tweets) > 0:
        # sum all the scores of each category
        features = describe_tweets(tweets)
    else:
        features = pd.DataFrame({'avg_len': 0, 'max_len': 0, 'min_len': 0,'avg_ret': 0, 'max_ret': 0, 'min_ret': 0, 'avg_fav': 0, 'max_fav': 0, 'min_fav': 0, 'avg_hash': 0, 'max_hash': 0, 'min_hash': 0, 'freq': 0, 'ret_perc': 0, 'media_perc': 0, 'url_perc': 0, 'quote_perc':0}, index=[0])
    
    # return the average scores of each user
    return features

features = describe(tweets)

# create dataset with all features

full = pd.concat([user_features, intradistances, scores, features, nsfw_data], axis=1)


def oldness(x):
    x = str(x)
    if x[0] == '2':
        return 2018 - int(x[:4])
    else:
        return 2018 - int(x[-4:])
    
full = full.drop(columns=['contributors_enabled', 'follow_request_sent', 'following', 'profile_background_image_url', 'profile_background_image_url_https', 'profile_image_url', 'profile_image_url_https', 'time_zone', 'utc_offset'])
full = full.drop(columns=['default_profile_image','is_translator', 'geo_enabled', 'location', 'notifications', 'profile_background_tile', 'protected'])
full['default_profile'] = full['default_profile'].apply(lambda x: int(x))
full['description'] = full['description'].fillna('')
full['description_len'] = full['description'].apply(lambda x: len(x))
full = full.drop(columns=['description'])
full['name'] = full['name'].fillna('')
full['name_len'] = full['name'].apply(lambda x: len(x))
full = full.drop(columns=['name'])
full['screen_name'] = full['screen_name'].fillna('')
full['screen_name_len'] = full['screen_name'].apply(lambda x: len(x))
full = full.drop(columns=['screen_name'])
full = full.drop(columns=['lang'])
full['age'] = full['created_at'].apply(lambda x: oldness(x))
full = full.drop(columns=['created_at'])
full = full.drop(columns=['id'])
full['profile_use_background_image'] = full['profile_use_background_image'].apply(lambda x: int(x))
full['url'] = (full['url'].notnull()).astype(int)

print(full)

# BoN classification

data = full[['statuses_count', 'followers_count', 'friends_count',
       'favourites_count', 'listed_count', 'url', 'default_profile',
       'profile_use_background_image', 'verified', 'avg_len', 'max_len',
       'min_len', 'avg_ret', 'max_ret', 'min_ret', 'avg_fav', 'max_fav',
       'min_fav', 'avg_hash', 'max_hash', 'min_hash', 'freq', 'ret_perc',
       'media_perc', 'url_perc', 'quote_perc', 'url_entropy',
       'tweet_intradistance', 'bot_words_score', 'gen_words_score',
       'description_len', 'name_len', 'screen_name_len','nsfw_profile']]

model = pickle.load(open('models/bon_refill.model', 'rb'))
data = data.reindex(sorted(data.columns), axis=1)
bon_pred = model.predict_proba(data)

# RF classification

data = full[['NSFW_words_score','news_spreaders_words_score','spam_bots_words_score','fake_followers_words_score',
             'avg_fav','avg_hash','avg_len','avg_ret','default_profile','favourites_count','followers_count',
             'freq','friends_count','listed_count','max_fav','max_hash','max_len','max_ret','media_perc','min_hash',
             'min_len','min_ret','nsfw_avg','nsfw_profile','profile_use_background_image','quote_perc','ret_perc',
             'statuses_count','tweet_intradistance','url','url_perc','description_len','name_len','screen_name_len',
             'age','url_entropy','min_fav']]

data = data.reindex(sorted(data.columns), axis=1)

model = pickle.load(open('models/rf.model', 'rb'))
rf_scores = model.predict_proba(data)
rf_prob = pd.DataFrame(rf_scores, columns=['rf_0', 'rf_1', 'rf_2', 'rf_3'])


## KNN classification

data = full[['default_profile', 'favourites_count', 'followers_count', 'friends_count', 'listed_count','profile_use_background_image',
       'statuses_count', 'url', 'description_len', 'name_len', 'screen_name_len', 'age']]
weights = pickle.load(open('models/knn.weights', 'rb'))
model = pickle.load(open('models/knn.model', 'rb'))

data = data * weights
knn_scores = model.predict_proba(data)
knn_prob = pd.DataFrame(knn_scores, columns=['knn_0', 'knn_1', 'knn_2', 'knn_3'])

# NB classification
if len(tweets) != 0:
    tweets = tweets['full_text']


    # initialize pipeline

    stemmer = SnowballStemmer("english", ignore_stopwords=True)

    class StemmedCountVectorizer(CountVectorizer):
        def build_analyzer(self):
            analyzer = super(CountVectorizer, self).build_analyzer()
            return lambda doc:(stemmer.stem(w) for w in analyzer(doc))


    # load model

    nb = pickle.load( open( "models/nb.model", "rb" ) )


    # define preprocessing functions

    def remove_rt(x):
        if 'RT @' in x:
            try:
                return x[x.find(':')+2:]
            except:
                return x
        else:
            return x


    stop_words = stopwords.words('english')

    def remove_stop(x):
        return [word for word in x.split() if word not in stop_words]


    # preprocess tweets

    tweets = tweets.apply(lambda x: remove_rt(x))
    tweets = tweets.apply(lambda x: re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', x))
    tweets = tweets.apply(lambda x: re.sub(r'[^\w\s]','',x))
    tweets = tweets.apply(lambda x: x.lower())
    tweets = tweets.apply(lambda x: remove_stop(x))
    tweets = tweets.astype(str)
    tweets = tweets[tweets!='[]']


    # perform predictions over tweets

    pred = nb.predict_proba(tweets)

    # return average of NB predictions

    nb_scores = np.mean(pred, axis=0)

else:
    nb_scores = np.array([0.25,0.25,0.25,0.25])

nb_scores = np.array(nb_scores)
nb_scores = np.expand_dims(nb_scores, axis=0)
nb_prob = pd.DataFrame(nb_scores, columns=['nb_0', 'nb_1', 'nb_2', 'nb_3'])


# merge predictions
prob = pd.concat([knn_prob,nb_prob,rf_prob], axis=1)


# LR classification

model = pickle.load(open("models/lr.model", "rb"))
mc = model.predict_proba(prob)


# compute final prediction

final = []
final.append(bon_pred[0][1] * mc[0][0])
final.append(bon_pred[0][1] * mc[0][1])
final.append(bon_pred[0][1] * mc[0][2])
final.append(bon_pred[0][1] * mc[0][3])
final.append(bon_pred[0][0])

print(final)
print('unreliability rate = ' + str(unreliability_rate))
