import tweepy
from textblob import TextBlob
import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt


import twitter_credentials

auth=tweepy.OAuthHandler(twitter_credentials.api_key,twitter_credentials.api_key_secret)
auth.set_access_token(twitter_credentials.access_token,twitter_credentials.access_token_secret)
twitterApi=tweepy.API(auth,wait_on_rate_limit=True)


twitterAccount="imVkohli"


tweets=tweepy.Cursor(twitterApi.user_timeline,
                    screen_name=twitterAccount,
                    count=None,
                    since_id=None,
                    max_id=None,trim_user=True,exclude_replies=True,contributer_details=False,
                    include_entities=False).items(50);

df=pd.DataFrame(data=[tweet.text for tweet in tweets],columns=['Tweet'])
df.head()

def cleanUpTweet(txt):
    txt=re.sub(r'@[A-Za-z0-9_]+','',txt)
    txt=re.sub(r'#','',txt)
    txt=re.sub(r'RT:','',txt)
    txt=re.sub(r'https?:\/\/[A-Za-z0-9\.\/]+','',txt)
    return txt
    
df['Tweet']=df['Tweet'].apply(cleanUpTweet)

def getTextSubjectivity(txt):
    return TextBlob(txt).sentiment.subjectivity

def getTextPolarity(txt):
    return TextBlob(txt).sentiment.polarity

df['Subjectivity']=df['Tweet'].apply(getTextSubjectivity)
df['Polarity']=df['Tweet'].apply(getTextPolarity)
df.head(50)

df=df.drop(df[df['Tweet']==''].index)

def getTextAnalysis(a):
    if a<0:
        return "Negative"
    if a==0:
        return "Neutral"
    else:
        return "Positive"

df["Score"]=df['Polarity'].apply(getTextAnalysis)
df.head(50)

positive=df[df['Score']=="Positive"]
print(str(positive.shape[0]/(df.shape[0])*100)+"% of positive tweets")
pos=positive.shape[0]/df.shape[0]*100

negative=df[df['Score']=="Negative"]
print(str(negative.shape[0]/(df.shape[0])*100)+"% of negative tweets")
neg=negative.shape[0]/df.shape[0]*100

neutral=df[df['Score']=="Neutral"]
print(str(neutral.shape[0]/(df.shape[0])*100)+"% of neutral tweets")
neutral=neutral.shape[0]/df.shape[0]*100

explode=(0,0.1,0)
labels='Positive','Negative','Neutral'
sizes=[pos,neg,neutral]
colors=['yellowgreen','lightcoral','gold']

plt.pie(sizes,explode=explode,colors=colors,autopct='%1.1f%%',startangle=120)
plt.legend(labels,loc=(-0.05,0.05),shadow=True)
plt.axis('equal')
plt.savefig('SentimentAnalysis.png')




