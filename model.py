import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow import keras
import yfinance as yf

startdate = '2014-06-30'
enddate = '2024-08-7'
data = yf.download('CIPLA.NS',start=startdate,end=enddate)['Close'].pct_change().dropna()

scaled = np.zeros(len(data))
for i in range(len(data)):
    scaled[i] = (data[i]-data.min())/(data.max()-data.min())

def sequences(seq_length, data):
    X = []
    y = []
    for i in range(len(data)-seq_length):
        X.append(data[i:i+seq_length])
        y.append(data[i+seq_length])
    return np.array(X), np.array(y)

X, y = sequences(60,scaled)

X_train, X_test = X[:int(0.8*len(X))], X[int(0.8*len(X)):]
y_train , y_test = y[:int(0.8*len(X))], y[int(0.8*len(X)):]

model = keras.Sequential()
model.add(keras.layers.LSTM(50,return_sequences=True,input_shape=(60,1)))
model.add(keras.layers.LSTM(50,return_sequences=True))
model.add(keras.layers.Conv1D(32,3))
model.add(keras.layers.Dropout(0.2))
model.add(keras.layers.Flatten())
model.add(keras.layers.Dense(100))
model.add(keras.layers.Dense(1))

model.compile(optimizer='adam', loss='mean_squared_error')
model.summary()

history = model.fit(X_train, y_train, epochs=40,batch_size=32,validation_split=0.2)

y_pred = model.predict(X_test)

print(np.sqrt(np.mean(y_test-y_pred)**2))

len(y_train)

def inverse(y):
    inv_scal = np.zeros(len(y))

    for i in range(len(y)):

        inv_scal[i] = y[i]*(data.max()-data.min()) + data.min()
    return inv_scal

y_test = inverse(y_test)
y_pred = inverse(y_pred)

plt.plot(y_pred,label= 'pred')
plt.plot(y_test, label='true')
plt.show()