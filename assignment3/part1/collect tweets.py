#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import tweepy
#conda install -c conda-forge tweepy
from tweepy import OAuthHandler


## All 4 keys are in my TwitterCodesFile.txt and are comma sep
filename="/Users/LiuYing/Desktop/GU/2021-summer/assignment/assignment3/part1/TwitterCodesFile.txt"
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


