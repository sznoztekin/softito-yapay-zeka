"""simple rnn.ipynb

"""

#pip install yfinance

import pandas as pd
import yfinance as yf

df = yf.download( "TSLA", period='5y')

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

#pip install tensorflow
import tensorflow as tf
from tensorflow import keras

df.info()

train_length = round(len(df)*0.7)
lg = len(df)
val_length = lg-train_length

print('Total observations:',lg)
print('Training set:', train_length)
print('Validation set:', val_length)

train_data = df[('Close', 'TSLA')][:train_length]
val_data = df[('Close', 'TSLA')][train_length:]

train=train_data.values.reshape(-1,1)
train

scaler = MinMaxScaler()
scaled_trainset = scaler.fit_transform(train)

x_train = []
y_train = []
step = 50

for i in range(step, train_length):
    x_train.append(scaled_trainset[i-step:i,0])
    y_train.append(scaled_trainset[i,0])

X_train, y_train = np.array(x_train), np.array(y_train)

X_train = np.reshape(X_train, (X_train.shape[0],X_train.shape[1],1))
y_train.reshape(y_train.shape[0],1)

import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import SimpleRNN
from keras.layers import Dropout

model = Sequential()

model.add(
    SimpleRNN(units = 50,return_sequences= True,input_shape = (X_train.shape[1],1)))

model.add(
    Dropout(0.2))

model.add(
    SimpleRNN(units = 50, return_sequences = True)
             )

model.add(
    Dropout(0.2)
             )

model.add(
    SimpleRNN(units = 50, return_sequences = True)
             )

model.add(
    Dropout(0.2)
             )

model.add(
    SimpleRNN(units = 50)
             )

model.add(
    Dropout(0.2)
             )

model.add(
    Dense(units = 1))

model.compile(optimizer = 'adam', loss = 'mean_squared_error', metrics = ['accuracy'])

model.summary()

history = model.fit(X_train, y_train, epochs = 10, batch_size =16)

y_pred = model.predict(X_train)
y_pred = scaler.inverse_transform(y_pred.reshape(1,-1))

X_train

y_train = scaler.inverse_transform(y_train.reshape(1,-1))
y_train

y_train = scaler.inverse_transform(y_train.reshape(1,-1))
y_train

y_train.shape
y_train = np.reshape(y_train, (828,1))

y_pred.shape

y_pred.shape
y_pred = np.reshape(y_pred,(828,1))
y_pred

val = val_data.values.reshape(-1,1)
val

scaled_valset = scaler.fit_transform(val)

xval_train = []
yval_train = []
step = 50

for i in range(step, val_length):
    xval_train.append(scaled_valset[i-step:i,0])
    yval_train.append(scaled_valset[i,0])

X_val, y_val = np.array(xval_train), np.array(yval_train)

X_val = np.reshape(X_val, (X_val.shape[0],X_val.shape[1],1))  # reshape to 3D array
y_val = np.reshape(y_val, (-1,1))

y_pred_val = model.predict(X_val)

y_val_is = scaler.inverse_transform(y_val)

import tensorflow as tf
import os
from tensorflow.keras.models import load_model

model.save(os.path.join('model', 'SimpleRNN_Forecasting.h5'))
new_model = load_model(os.path.join('model', 'SimpleRNN_Forecasting.h5'))