import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
import seaborn as sns

startdate = '2023-06-30'
enddate = '2024-07-30'

tickers=['TCS.NS','HCLTECH.NS','WIPRO.NS','TECHM.NS','INFY.NS','LTTS.NS','KPITTECH.NS','COFORGE.NS','PERSISTENT.NS','MPHASIS.NS','LTIM.NS','AFFLE.NS','FSL.NS','LICI.NS']

data = yf.download(tickers, start=startdate,end=enddate)['Close']
data.pct_change().dropna()
correlation_mat = data.corr()

correlation_mat







