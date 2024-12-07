import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt


# gather the instrument data

startdate = '2010-06-30'
enddate = '2024-07-30'
data = yf.download('INFY.NS',start=startdate,end=enddate,interval='1d')

# calculate the pct change of closing prices in daily interval
data = data['Close'].pct_change().to_numpy()
data *= 100
data = data[1:]
data[-1]

data

#normalize the data
for i in range(len(data)):

    data[i] = (data[i] - data.min())/(data.max()-data.min())
data.min()
data.max()

#define the state space
n_states = 10
regimes = np.linspace(0,1,n_states)
len(regimes)

# map the data on to the statespace
for i in range(len(data)):

    for j in range(len(regimes)-1):

        if data[i] < regimes[j] and data[i]< regimes[j+1]:

            data[i] = j

data.max()

# calculate the transition counts
counts = np.zeros((n_states-1,n_states-1))
for (i, j) in zip(data[:-1],data[1:]):
    i= int(i)
    j = int(j)
    counts[i,j] += 1
counts


# normalize and calculate the transition matrix
for i in range(n_states-1):
    counts[i,:] /= np.sum(counts[i,:])

transition_matrix = counts

transition_matrix

plt.plot(transition_matrix[2])
plt.plot(transition_matrix[4])
plt.show()
#calculate the stationary distribution
for i in range(20):

    counts = counts@transition_matrix

stationary_dist = counts

plt.plot(stationary_dist[0])
plt.show()

counts
#plt.plot(transition_matrix[0])
plt.plot(transition_matrix[4])
plt.show()
