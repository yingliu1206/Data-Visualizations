#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import tweepy
#conda install -c conda-forge tweepy
from tweepy import OAuthHandler
#Loading libraries
import re
import pandas as pd # for data munging, it contains manipulation tools designed to make data analysis fast and easy

# For viz
import matplotlib.pyplot as plt
import seaborn as sns
from plotly import graph_objs as go
import plotly.express as px
from collections import Counter
from sklearn.feature_extraction.text import CountVectorizer
from wordcloud import WordCloud, STOPWORDS


## All 4 keys are in my TwitterCodesFile.txt and are comma sep
filename="/Users/LiuYing/Desktop/GU/2021-summer/assignment/assignment3/TwitterCodesFile.txt"
with open(filename, "r") as FILE:
    keys=[i for line in FILE for i in line.split(',')]
    
#API Key:
consumer_key = keys[0]
#API Secret Key:
consumer_secret =keys[1]
#Access Token:
access_token =keys[2]
#Access Token Secret:
access_secret =keys[3]


auth = OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_secret)
 
api = tweepy.API(auth)

# Define the search term and the date_since date as variables
search_words = "#covid19"
date_since = "2020-01-04"

# Collect tweets
tweets = tweepy.Cursor(api.search,
              q=search_words,
              lang="en",
              since=date_since).items(5)
tweets

## Create a new csv file to save the output
MyFILE=open("covid19.csv","w")
for tweet in tweets:
    print(tweet.text)
MyFILE.close()

df = pd.read_csv('/Users/LiuYing/Desktop/GU/2021-summer/assignment/assignment3/covid19.csv',encoding='latin1')

#################### data processing ####################
## Tokenize
from nltk.tokenize import RegexpTokenizer
tokenizer = RegexpTokenizer(r'\w+')
words_descriptions = df['text'].apply(tokenizer.tokenize)
words_descriptions.head()

## Vectorize
textlist=[]
for next1 in df['text']:
    next1=re.sub(r'[,.;@#?!&$\-\']+', ' ', next1, flags=re.IGNORECASE)
    next1=re.sub(r'\ +', ' ', next1, flags=re.IGNORECASE)
    next1=re.sub(r'\"', ' ', next1, flags=re.IGNORECASE)
    next1=re.sub(r'[^a-zA-Z]', " ", next1, flags=re.VERBOSE)
    textlist.append(next1)

MyCountV=CountVectorizer(input="content", lowercase=True, stop_words = "english")
 
MyDTM = MyCountV.fit_transform(textlist)  # create a sparse matrix
print(type(MyDTM))
#vocab is a vocabulary list
vocab = MyCountV.get_feature_names()  # change to a list
print(list(vocab)[10:20])

MyDTM = MyDTM.toarray()  # convert to a regular array
print(type(MyDTM))

ColumnNames=MyCountV.get_feature_names()
MyDTM_DF=pd.DataFrame(MyDTM,columns=ColumnNames)
print(MyDTM_DF)

all_words = [word for tokens in words_descriptions for word in tokens]
df['description_lengths']= [len(tokens) for tokens in words_descriptions]
VOCAB = sorted(list(set(all_words)))
print("%s words total, with a vocabulary size of %s" % (len(all_words), len(VOCAB)))

# Checking most common words
count_all_words = Counter(all_words)
count_all_words.most_common(50)

###################### EDA ######################
## common words
df['temp_list'] = df['text'].apply(lambda x:str(x).split())
top = Counter([item for sublist in df['temp_list'] for item in sublist])
temp = pd.DataFrame(top.most_common(20))
temp.columns = ['Common_words','count']
fig = px.bar(temp, x="count", y="Common_words", title='Commmon Words in Selected Text', orientation='h', 
             width=700, height=700,color='Common_words')
fig.show()

# Tree of the most common words
fig = px.treemap(temp, path=['Common_words'], values='count',title='Tree of Most Common Words')
fig.show()

# wordcloud1
comment_words = '' 
stopwords = set(STOPWORDS) 
  
# iterate through the csv file 
for val in df.text: 
      
    # typecaste each val to string 
    val = str(val) 
  
    # split the value 
    tokens = val.split() 
      
    # Converts each token into lowercase 
    for i in range(len(tokens)): 
        tokens[i] = tokens[i].lower() 
      
    comment_words += " ".join(tokens)+" "
  
wordcloud = WordCloud(width = 800, height = 800, 
                background_color ='white', 
                stopwords = stopwords, 
                min_font_size = 10).generate(comment_words) 
  
# plot the WordCloud image                        
plt.figure(figsize = (8, 8), facecolor = None) 
plt.imshow(wordcloud) 
plt.axis("off") 
plt.tight_layout(pad = 0) 
  
plt.show() 

#################### analysis ####################
##Applying VADER
# calling SentimentIntensityAnalyzer object
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
analyser = SentimentIntensityAnalyzer()

# Using polarity scores for knowing the polarity of each text
def sentiment_analyzer_score(sentence):
    score = analyser.polarity_scores(sentence)
    print("{:-<40} {}".format(sentence, str(score)))
    
#testing the function
tweet  = "I would love to watch the magic show again"
tweet2 = "What the hell they have made. Pathetic!"
tweet3 = " I do not know what to do"  
print (sentiment_analyzer_score(tweet))
print (sentiment_analyzer_score(tweet2))
print (sentiment_analyzer_score(tweet3))

## for the covid19
df['scores'] = df['text'].apply(lambda review: analyser.polarity_scores(review))
df['compound']  = df['scores'].apply(lambda score_dict: score_dict['compound'])
def Sentimnt(x):
    if x>= 0.05:
        return "Positive"
    elif x<= -0.05:
        return "Negative"
    else:
        return "Neutral"
#df['Sentiment'] = df['compound'].apply(lambda c: 'positive' if c >=0.00  else 'negative')
df['Sentiment'] = df['compound'].apply(Sentimnt)

var1 = df.groupby('Sentiment').count()['text'].reset_index().sort_values(by='text',ascending=False)

## p1
sns.set_style("white")
sns.set_palette("Set2")
var1.style.background_gradient()
plt.figure(figsize=(12,6))
sns.countplot(x='Sentiment',data=df)

## p2
fig = go.Figure(go.Funnelarea(
    text =var1.Sentiment,
    values = var1.text,
    title = {"position": "top center", "text": "Funnel-Chart of Sentiment Distribution"}
    ))
fig.show()

## positive wordcloud
df_positive = df[df["Sentiment"]== "Positive"] 
# iterate through the csv file 
for val in df_positive.text: 
      
    # typecaste each val to string 
    val = str(val) 
  
    # split the value 
    tokens = val.split() 
      
    # Converts each token into lowercase 
    for i in range(len(tokens)): 
        tokens[i] = tokens[i].lower() 
      
    comment_words += " ".join(tokens)+" "
  
wordcloud = WordCloud(width = 800, height = 800, 
                background_color ='white', 
                stopwords = stopwords, 
                min_font_size = 10).generate(comment_words) 
  
# plot the WordCloud image                        
plt.figure(figsize = (8, 8), facecolor = "green") 
plt.imshow(wordcloud) 
plt.axis("off") 
plt.tight_layout(pad = 0) 
  
plt.show() 

#negative wordcloud
df_negative = df[df["Sentiment"]== "Negative"] 
# iterate through the csv file 
for val in df_negative.text: 
      
    # typecaste each val to string 
    val = str(val) 
  
    # split the value 
    tokens = val.split() 
      
    # Converts each token into lowercase 
    for i in range(len(tokens)): 
        tokens[i] = tokens[i].lower() 
      
    comment_words += " ".join(tokens)+" "
  
wordcloud = WordCloud(width = 800, height = 800, 
                background_color ='white', 
                stopwords = stopwords, 
                min_font_size = 10).generate(comment_words) 
  
# plot the WordCloud image                        
plt.figure(figsize = (8, 8), facecolor = "red") 
plt.imshow(wordcloud) 
plt.axis("off") 
plt.tight_layout(pad = 0) 
  
plt.show() 

# netural wordcloud
comment_words = '' 
stopwords = set(STOPWORDS) 
  
df_neutral = df[df["Sentiment"]== "Neutral"] 
# iterate through the csv file 
for val in df_positive.text: 
      
    # typecaste each val to string 
    val = str(val) 
  
    # split the value 
    tokens = val.split() 
      
    # Converts each token into lowercase 
    for i in range(len(tokens)): 
        tokens[i] = tokens[i].lower() 
      
    comment_words += " ".join(tokens)+" "
  
wordcloud = WordCloud(width = 800, height = 800, 
                background_color ='white', 
                stopwords = stopwords, 
                min_font_size = 10).generate(comment_words) 
  
# plot the WordCloud image                        
plt.figure(figsize = (8, 8), facecolor = "yellow") 
plt.imshow(wordcloud) 
plt.axis("off") 
plt.tight_layout(pad = 0) 
  
plt.show() 
