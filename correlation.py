import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
import seaborn as sns

startdate = '2014-06-30'
enddate = '2024-07-30'

tickers=['SUNPHARMA.NS','DIVISLAB.NS','CIPLA.NS','TORNTPHARM.NS','DRREDDY.NS','ZYDUSLIFE.NS','LUPIN.NS','MANKIND.NS','AUROPHARMA.NS','ALKEM.NS']

data = yf.download(tickers, start=startdate,end=enddate)['Close']
data.pct_change().dropna()
correlation_mat = data.corr()

sns.heatmap(correlation_mat,annot=True)
plt.show()







