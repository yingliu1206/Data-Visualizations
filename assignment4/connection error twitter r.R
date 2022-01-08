consumer_key<-"0FxprV6cqnSUSq6yIqMKxSlL9"
consumer_secret<-"yD2dCfiTL7Vef0DUnvYEakca7jbBHX0F9aDcHQu70F4wYDB6ER"
access_token <- "1401013976184475652-IQDA1ope2fcQBsolabsI6eYx2qh0re"
access_secret <- "MuFnFja0gMabLKiuUukCWQjqO9mdHyrKaCLjjAz0vX6Rr"

packages <- c("twitteR", "openssl")
### checking if packages are already installed and installing if not
for(i in packages){
  if(!(i %in% installed.packages()[, "Package"])){
    install.packages(i)
  }
  library(i, character.only = TRUE) ## load packages
}

setup_twitter_oauth(consumer_key, consumer_secret)

Search_DF = read.csv("/Users/LiuYing/Desktop/GU/2021-summer/assignment/assignment4/vaccination_tweets.csv")
Search_DF = Search_DF[1:500,]
Search_DF$text = as.character(Search_DF$text)
