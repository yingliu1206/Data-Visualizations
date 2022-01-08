#!/usr/bin/env python3
# -*- coding: utf-8 -*-

###Packages-----------------------
import pandas as pd
import tweepy
#conda install -c conda-forge tweepy
from tweepy import OAuthHandler
import json
from tweepy import Stream
from tweepy.streaming import StreamListener
import sys

from nltk.tokenize import word_tokenize
from nltk.tokenize import TweetTokenizer
import re

from os import path
#from scipy.misc import imread
import matplotlib.pyplot as plt
##install wordcloud
## conda install -c conda-forge wordcloud
## May also have to run conda update --all on cmd
#import PIL
#import Pillow
#import wordcloud
from wordcloud import WordCloud, STOPWORDS
###-----------------------------------------


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

##-----------------------------------------------------------------
#Other Tweepy options - FYI
#for status in tweepy.Cursor(api.home_timeline).items(10):
#Process a single status
 #   print(status.text) 
#   
#def Gather(tweet):
 #   print(json.dumps(tweet))
#for friend in tweepy.Cursor(api.friends).items():
 #   Gather(friend._json)
#-------------------------------------------------------------- 
class Listener(StreamListener):
    print("In Listener...") 
    tweet_number=0
    #__init__ runs as soon as an instance of the class is created
    def __init__(self, max_tweets, hfilename, rawfile):
        self.max_tweets=max_tweets
        print(self.max_tweets)     
    #on_data() is a function of StreamListener as is on_error and on_status    
    def on_data(self, data):
        self.tweet_number+=1 
        print("In on_data", self.tweet_number)
        try:
            print("In on_data in try")
            with open(hfilename, 'a') as f:
                with open(rawfile, 'a') as g:
                    tweet=json.loads(data)
                    tweet_text=tweet["text"]
                    print(tweet_text,"\n")
                    f.write(tweet_text) # the text from the tweet
                    json.dump(tweet, g)  #write the raw tweet
        except BaseException:
            print("NOPE")
            pass
        if self.tweet_number>=self.max_tweets:
            #sys.exit('Limit of '+str(self.max_tweets)+' tweets reached.')
            print("Got ", str(self.max_tweets), "tweets.")
            return False
    #method for on_error()
    def on_error(self, status):
        print("ERROR")#machi
        print(status)   #401 your keys are not working
        if(status==420):
            print("Error ", status, "rate limited")
            return False
#----------------end of class Listener
        
hashname=input("Enter the hash name, such as #womensrights: ") 
numtweets=eval(input("How many tweets do you want to get?: "))
if(hashname[0]=="#"):
    nohashname=hashname[1:] #remove the hash
else:
    nohashname=hashname
    hashname="#"+hashname

#Create a file for any hash name    
hfilename="file_"+nohashname+".csv"
rawfile="file_rawtweets_"+nohashname+".csv"
twitter_stream = Stream(auth, Listener(numtweets, hfilename, rawfile))
#twitter_stream.filter(track=['#womensrights'])
twitter_stream.filter(track=[hashname])
print("Twitter files created....")

#-----------------------------------
#-----------------------------------
#https://docs.python.org/3/library/re.html

linecount=0
hashcount=0
wordcount=0
BagOfWords=[]
BagOfHashes=[]
BagOfLinks=[]

### SET THE FILE NAME ###

tweetsfile=hfilename

###################################

with open(tweetsfile, 'r') as file:
    for line in file:
        #print(line,"\n")
        tweetSplitter = TweetTokenizer(strip_handles=True, reduce_len=True)
        WordList=tweetSplitter.tokenize(line)
        #WordList2=word_tokenize(line)
        #linecount=linecount+1
        #print(WordList)
        #print(len(WordList))
        #print(WordList[0])
        #print(WordList2)
        #print(len(WordList2))
        #print(WordList2[3:6])
        #print("NEXT..........\n")
        regex1=re.compile('^#.+')
        regex2=re.compile('[^\W\d]') #no numbers
        regex3=re.compile('^http*')
        regex4=re.compile('.+\..+')
        for item in WordList:
            if(len(item)>2):
                if((re.match(regex1,item))):
                    #print(item)
                    newitem=item[1:] #remove the hash
                    BagOfHashes.append(newitem)
                    hashcount=hashcount+1
                elif(re.match(regex2,item)):
                    if(re.match(regex3,item) or re.match(regex4,item)):
                        BagOfLinks.append(item)
                    else:
                        BagOfWords.append(item)
                        wordcount=wordcount+1
                else:
                    pass
            else:
                pass
            
    
        
       
#print(linecount)            
#print(BagOfWords)
#print(BagOfHashes)
#print(BagOfLinks)
BigBag=BagOfWords+BagOfHashes




#list of words I have seen
seenit=[]
#dict of word counts
WordDict={}
Rawfilename="TwitterResultsRaw.txt"
Freqfilename="TwitterWordFrq.txt"


#FILE=open(Freqfilename,"w")
#FILE2=open(Rawfilename, "w")
R_FILE=open(Rawfilename,"w")
F_FILE=open(Freqfilename, "w")

IgnoreThese=["and", "And", "AND","THIS", "This", "this", "for", "FOR", "For", 
             "THE", "The", "the", "is", "IS", "Is", "or", "OR", "Or", "will", 
             "Will", "WILL", "God", "god", "GOD", "Bible", "bible", "BIBLE",
             "CanChew", "Download", "free", "FREE", "Free", "will", "WILL", 
             "Will", "hits", "hit", "within", "steam", "Via", "via", "know", "Study",
             "study", "unit", "Unit", "always", "take", "Take", "left", "Left",
             "lot","robot", "Robot", "Lot", "last", "Last", "Wonder", "still", "Still",
             "ferocious", "Need", "need", "food", "Food", "Flint", "MachineCredit",
             "Webchat", "luxury", "full", "fifdh17", "New", "new", "Caroline",
             "Tirana", "Shuayb", "repro", "attempted", "key", "Harrient", 
             "Chavez", "Women", "women", "Mumsnet", "Ali", "Tubman", "girl","Girl",
             "CSW61", "IWD2017", "Harriet", "Great", "great", "single", "Single", 
             "tailoring", "ask", "Ask"]
###Look at the words
for w in BigBag:
    if(w not in IgnoreThese):
        rawWord=w+" "
        R_FILE.write(rawWord)
        if(w in seenit):
            #print(w, seenit)
            WordDict[w]=WordDict[w]+1 #increment the times word is seen
        else:
            ##add word to dict and seenit
            seenit.append(w)
            WordDict[w]=1
    
#print(WordDict)  
#print(seenit)
#print(BagOfWords)



for key in WordDict:
    #print(WordDict[key])
    if(WordDict[key]>1):
        if(key not in IgnoreThese):
            #print(key)
            Key_Value=key + "," + str(WordDict[key]) + "\n"
            F_FILE.write(Key_Value)


#FILE.close()
#FILE2.close()
R_FILE.close()
F_FILE.close()

#----------------------------
#------------------------------
d = path.dirname(__file__)
Rawfilename="TwitterResultsRaw.txt"

# Read the whole text.
text = open(path.join(d, Rawfilename)).read()
##print(text)
## --OR --
##with open("constitution.txt") as f:
##    lines f.readlines()                                                                            
##text = "".join(lines) 
##---------
wordcloud = WordCloud().generate(text)
# Open a plot of the generated image.
#figure(figsize = (20,2))
plt.figure(figsize=(50,40))
plt.imshow(wordcloud)
           #, aspect="auto")
plt.axis("off")
##trumpplt.show()
