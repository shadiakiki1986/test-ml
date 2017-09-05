# libraries (copy from p5g)

# import os
import numpy as np
from keras.models import Sequential
from keras.layers.core import Dense, Dropout
from keras.layers.recurrent import SimpleRNN, LSTM, GRU
from keras.callbacks import Callback, EarlyStopping, ModelCheckpoint
import keras
import utils
import utils2

import pandas as pd

#-----------------------------------
# copy from utils3.py

# stride the data
from skimage.util.shape import view_as_windows # pip install scikit-image
def _load_data_strides(A, n_prev):
    out = view_as_windows(A,window_shape=(n_prev,A.shape[1]),step=1)
    out = out.reshape((out.shape[0],out.shape[2],out.shape[3])) # for some reason need to drop extra dim=1
    return out

#-----------------------------------
from keras.models import load_model
from shutil import copyfile
from datetime import datetime

from keras.layers.recurrent import LSTM
from keras.models import Model
from keras.models import Sequential
from keras.layers.core import Dense

# simulated data (copy from p5g)
# nb_samples = int(1e3)
def data(nb_samples:int):
  if nb_samples<=0: raise Exception("nb_samples <= 0")

  lags = [1, 2]
  X1 = pd.Series(np.random.randn(nb_samples))
  X2 = pd.Series(np.random.randn(nb_samples))
  # https://stackoverflow.com/a/20410720/4126114
  X_model = pd.concat({'main': X1, 'lagged 1': X1.shift(lags[0]), 'lagged 2': X1.shift(lags[1]), 'new': X2}, axis=1).dropna()
                       
  X_model['mult'] = X_model.apply(lambda row: row[2]*row[3], axis=1)
  
  
  # Y = X_model.apply(lambda row: 0.25*row[0] + 0.25*row[1] + 0.25*row[2] + 0.25*row[3], axis=1)
  
  Y = X_model.apply(lambda row: 0.2*row['main'] + 0.2*row['lagged 1'] + 0.2*row['lagged 2'] + 0.2*row['new'] + 0.2*row['mult'], axis=1)
  Y = Y.reshape((Y.shape[0],1))

  # drop columns in X_model that LSTM is supposed to figure out
  del X_model['lagged 1']
  del X_model['lagged 2']
  del X_model['mult']

  return (X_model, Y, lags)

# copy from g2-ml/take2/ex5-lstm/p4c4.ipynb
# lstm_dim = 30
#  in_neurons = X_model.shape[1]
def model(in_neurons:int, lstm_dim:list):  
  if len(lstm_dim)==0: raise Exception("len(lstm_dim) == 0")

  optimizer='adam'
  
  model = Sequential()
  for i, dimx in enumerate(lstm_dim):
    if i==0:
      # return sequences: for multi-stack, all True except last should be False
      model.add(LSTM(dimx, return_sequences=len(lstm_dim)!=1, input_shape=(None, in_neurons), activation='tanh'))#, dropout=0.25))
    else:
      model.add(LSTM(dimx, return_sequences=i!=(len(lstm_dim)-1), activation='tanh'))
  out_neurons = 1
  model.add(Dense(out_neurons, activation='linear'))
  
  model.compile(loss="mean_squared_error", optimizer=optimizer) # nadam

  return model

#  epochs = 300
#  look_back = 5
def fit(X_model:pd.DataFrame, Y, lags:list, model, epochs:int, look_back:int):
  if epochs<=0: raise Exception("epochs <= 0")

  if look_back < max(lags):
      raise Exception("Not enough look back provided")
  X_calib = _load_data_strides(X_model.values, look_back)
  
  Y_calib = Y[(look_back-1):]
    
  history = model.fit(
      x=X_calib,
      y=Y_calib,
      epochs = epochs,
      verbose = 0, #2,
      batch_size = 1000, # 100
      validation_split = 0.2,
      shuffle=False
  )
  
  pred = model.predict(x=X_calib, verbose = 0)
  
  err = utils.mse(Y_calib, pred)

  return (history, err)
