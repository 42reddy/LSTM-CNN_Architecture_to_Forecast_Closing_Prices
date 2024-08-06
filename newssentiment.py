import numpy as np
import requests
from newspaper import Article
import matplotlib.pyplot as plt


from GoogleNews import GoogleNews

news = GoogleNews()
news.setlang('en')
news.set_period('7d')
news.search('INFY')

result = news.result()
links = []
for article in result:
    links.append(article['link'])
print(links)

from newsapi import  NewsApiClient

session = NewsApiClient(api_key='55ce817a953a4dff9a286626fb766511')
articles = session.get_everything(q='gaza war', sort_by='relevancy')
articles

from textblob import TextBlob

url = 'https://finance.yahoo.com/news/indias-hcltech-q1-revenue-line-123123148.html'


def extract(url):
    article = Article(url)
    article.download()
    article.parse()
    return article.text
news = []

#extract('https://www.businesstoday.in/markets/stocks/story/infosys-shares-in-focus-amid-rs-32403-crore-gst-demand-notice-what-it-firm-says-439648-2024-08-01')


#top_news = session.get_everything(q='semiconductor industry',sort_by='relevancy')
#top_news
x = []
for i in articles['articles']:
    x.append(i['description'])
x
y=[]

for i in range(15):

    y.append(TextBlob(x[i]).sentiment.polarity)
y = np.array(y)
y
print(y.mean())

TextBlob(x[4]).sentiment
print(x[14])

for i in articles['articles']:
    news.append(extract(i['url']))
news
for i in range(len(news)):

    y.append(TextBlob(news[i]).sentiment.polarity)

y