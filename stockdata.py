import numpy as np
import pandas as pd
from datetime import date
from sklearn.svm import SVR

import yfinance as yf

START = "2022-01-01"
TODAY = date.today().strftime("%Y-%m-%d")


def getStockData(start,end,ticker):
    stock = ticker
    data = yf.download(stocks[0], START, TODAY)
    data.reset_index(inplace=True)
    # data['date'] = data['Date']
    for i in range(10):
        data["price lag-" + str(i+1)]=data["Close"].shift(i+1)
    data['MA5'] = data['Close'].rolling(5, closed='left').mean()
    data['MA10'] = data['Close'].rolling(10, closed='left').mean()
    data['MA20'] = data['Close'].rolling(20, closed='left').mean()
    data['MA60'] = data['Close'].rolling(60, closed='left').mean()
    data = data.dropna(inplace=True)
    return data

# print(getStockData(START,TODAY,'BBCA.JK'))

import snscrape.modules.twitter as sntwitter
from textblob import TextBlob

def getPolarity(text):
    polarity = TextBlob(text).sentiment.polarity
    return int(polarity*1000)/1000

def getSentiment(polarity):
    # polarity = TextBlob(text).sentiment.polarity
    if polarity>0:
      sent = "Positive"
    elif polarity<0:
      sent ="Negative"
    else:
      sent ="Neutral"
    return sent

def getStockSentiment(ticker,from_date,end_date):
    tweets_list2 = []
    # Using TwitterSearchScraper to scrape data and append tweets to list
    for i,tweet in enumerate(sntwitter.TwitterSearchScraper(f'{ticker} since:{from_date} until:{end_date}').get_items()):
        if i>100:
            break
        tweets_list2.append([tweet.date, tweet.id, tweet.content, tweet.username])
        # tweet.retweetCount
    df3 = pd.DataFrame(tweets_list2, columns=['Datetime', 'Tweet Id', 'Text', 'Username'])
    # df3['Sentiment'] = getSentimentVader(df3['Text'].tolist())
    df3['Polarity'] = df3['Text'].apply(getPolarity)
    df3['Sentiment'] = df3['Polarity'].apply(getSentiment)
    df3 = df3[['Datetime','Sentiment', 'Polarity','Username','Text']]
    return df3

def getDailysentiment(data):
    df = data
    df['date'] = df['Datetime'].dt.date
    df = df.groupby(by=['date'],as_index=False).agg({'Polarity':'sum','Sentiment':'count'})
    return df


df = getStockSentiment('ADARO',START,TODAY)
print(df.tail(5))

print(getDailysentiment(df))
