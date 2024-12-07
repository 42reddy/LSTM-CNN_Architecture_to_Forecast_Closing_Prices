
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import STL
from tensorflow import keras


class emdlstm():

    def __init__(self,startdate,enddate,seq_length):
        self.startdate = startdate
        self.enddate = enddate
        self.seq_length = seq_length

    """take in a time series and returns three decomposed series describing the former"""
    def stl(self,data):
        result = STL(data,period=2)
        result = result.fit()
        return result.trend, result.seasonal, result.resid

    """fetches the closing price data of the instrument in one day interval"""
    def get_data(self,ticker):
        return yf.download(ticker,start=self.startdate,end =self.enddate,interval='1d')['Close']

    """takes in a time series and retuns the sequences of data suitable for an RNN"""
    def sequences(self,data):
        X = []
        y = []
        for i in range(len(data) - self.seq_length):
            X.append(data[i:i + self.seq_length])
            y.append(data[i+self.seq_length])
        return np.array(X), np.array(y)

    """splits the imput data into training and test sets"""
    def split(self,X,y):
        X_train, X_test = X[:int(0.9 * len(X))], X[int(0.9 * len(X)):]
        y_train, y_test = y[:int(0.9 * len(X))], y[int(0.9 * len(X)):]

        return X_train, X_test, y_train, y_test

    """translates the normalized predicted values back to original scale"""
    def inverse(self,y,data):
        inv_scal = np.zeros(len(y))
        for i in range(len(y)):
            inv_scal[i] = y[i] * (data.max() - data.min()) + data.min()
        return inv_scal

    def smoothen(self,data):
        data_hat = np.fft.fft(data)
        power = data_hat*np.conj(data_hat)/(len(data)-1)
        ind = power > 0.005*power.mean()
        data = data_hat*ind
        return np.real(np.fft.ifft(data))

    def add_noise(self,data,n):
        for i in range(len(data)):
            data[i] += n*np.random.rand()
        return data

    def exponential(self,data,smoothing):
        data_exp = np.zeros(len(data))
        data_exp[0] = data[0]
        for i in range(len(data)-1):
            data_exp[i+1] = (data[i+1]*(smoothing/(1+len(data)))) + data_exp[i]*(1-(smoothing/(1+len(data))))
        return data_exp

    def moving_avg(self,data):
        mvn_avg = np.zeros_like(data)
        mvn_avg[0] = scaled_data[0]
        for i in range(len(data)-1):
            mvn_avg[i+1] = (mvn_avg[i]*(i+1)+data[i+1])/(i+1)
        return mvn_avg


start_date = '2022-06-30'
end_date = '2023-08-22'

instance = emdlstm(startdate=start_date,enddate=end_date,seq_length=20)  # initiate the class
dji = 'PERSISTENT.NS'                   # ticker symbol for s&p 500
volume = yf.download('PERSISTENT.NS', start=start_date, end=end_date)['Volume']
volume = (volume - np.mean(volume)) / np.std(volume)

data = instance.get_data(dji) # fetch the closing prices

scaled_data = (data - np.mean(data)) / np.std(data)  # normalize the series

trend, seasonal, resid = instance.stl(scaled_data)   # decompose the time series
trend_seq, _ = instance.sequences(trend)      # generate sequnces from the decomposed time series
seasonal_seq, _ = instance.sequences(seasonal)
resid_seq, _ = instance.sequences(resid)
volume_seq, _ = instance.sequences(volume)

X = np.stack((trend_seq, seasonal_seq, resid_seq, volume_seq),axis=-1)
# X = np.stack((trend_seq,seasonal_seq,resid_seq,trend_indseq,seasonal_indseq,resid_indseq),axis=-1)       # prepare the input suitable to train the model
_,y = instance.sequences(scaled_data)


"""build the CNN-LSTM network"""
"""
input = keras.layers.Input((60,3))
conv = keras.layers.Conv1D(64,5)(input)
conv = keras.layers.MaxPool1D(2)(conv)
lstm = keras.layers.LSTM(200,
                         'tanh',return_state=True,return_sequences=True)(conv)
#attention = keras.layers.Attention()(lstm)
lstm = keras.layers.Flatten(lstm)
output = keras.layers.Dense(1,activation='linear',trainable=True)(lstm)

model = keras.Model(input,output)

"""
model = keras.Sequential()
model.add(keras.layers.Conv1D(16,5))
model.add(keras.layers.MaxPool1D(2))
#model.add(keras.layers.Dropout(0.2))
model.add(keras.layers.LSTM(250,return_sequences=True,input_shape=(20,4)))
model.add(keras.layers.Flatten())
model.add(keras.layers.Dense(32,activation='relu'))
model.add(keras.layers.Dense(1))


model.compile(keras.optimizers.legacy.Adam(), loss='mean_squared_error')


history = model.fit(X,y,epochs=80,validation_split=0.2)    # train the model
plt.plot(history.history['loss'],label='train loss')
plt.plot(history.history['val_loss'],label='val_loss')
plt.legend()
plt.show()

#mymodel = keras.models.load_model('s&p.h5')


start_date = '2023-09-23'
end_date = '2024-11-20'
instance = emdlstm(startdate=start_date,enddate=end_date,seq_length=20)
xtest = instance.get_data('INFY.NS')
volume = yf.download('INFY.NS', start=start_date, end=end_date)['Volume']
volume = (volume - np.mean(volume)) / np.std(volume)
#xtest = instance.smoothen(xtest)
scaled_xtest = (xtest - np.mean(xtest)) / np.std(xtest)
trend_xtest, seasonal_xtest, resid_xtest = instance.stl(scaled_xtest)
trend_seqxtest, _ = instance.sequences(trend_xtest)      # generate sequnces from the decomposed time series
seasonal_seqxtest, _ = instance.sequences(seasonal_xtest)
resid_seqxtest, _ = instance.sequences(resid_xtest)
volume_seq, _ = instance.sequences(volume)


Xtest = X = np.stack((trend_seqxtest, seasonal_seqxtest, resid_seqxtest, volume_seq),axis=-1)
_,ytest = instance.sequences(scaled_xtest)
y_pred = model.predict(Xtest)

y_pred = instance.inverse(y_pred,xtest)           # revert back to original scale
ytest = instance.inverse(ytest,xtest)
print(np.sqrt(np.sum((ytest-y_pred)**2)/len(xtest)))
print(np.sqrt(np.sum((ytest-y_pred)**2)/len(xtest))/np.mean(xtest))
print(np.mean(xtest))
plt.plot(y_pred, label = 'predicted value')
plt.plot(ytest, label= 'true value')
plt.title('HCLTECH true vs predicted values')
plt.legend()
plt.show()

plt.plot(ytest - y_pred)




