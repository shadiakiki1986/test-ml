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


# copy from g2-ml/take2/ex5-lstm/p4c4.ipynb
# lstm_dim = 30
#  in_neurons = X_model.shape[1]
def model(in_neurons:int, lstm_dim:list, look_back:int):
  if len(lstm_dim)==0: raise Exception("len(lstm_dim) == 0")

  optimizer='adam'
  out_neurons = 1
 
  model = Sequential()
  for i, dimx in enumerate(lstm_dim):
    print(i, dimx, len(lstm_dim))
    if i==0:
      # return sequences: for multi-stack, all True except last should be False
      model.add(LSTM(dimx, return_sequences=len(lstm_dim)!=1, input_shape=(None, in_neurons), activation='tanh'))#, dropout=0.25))
    else:
      model.add(LSTM(dimx, return_sequences=(i+1)!=len(lstm_dim), activation='tanh'))

  model.add(Dense(out_neurons, activation='linear'))
  
  model.compile(loss="mean_squared_error", optimizer=optimizer) # nadam

  return model

from keras.layers import RepeatVector, TimeDistributed, Input

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
      verbose = 2,
      batch_size = 1000, # 100
      validation_split = 0.2,
      shuffle=False
  )
  
  pred = model.predict(x=X_calib, verbose = 0)
  
  err = utils.mse(Y_calib, pred)

  return (history, err)


# -----------------------------
# copy from g2-ml/take2/ex5-lstm/p4c4.ipynb
# lstm_dim = 30
#  in_neurons = X_model.shape[1]
def model_2(in_neurons:int, lstm_dim:list, look_back:int):
  if len(lstm_dim)==0: raise Exception("len(lstm_dim) == 0")

  optimizer='adam'
  out_neurons = 1
 
  model = Sequential()
  for i, dimx in enumerate(lstm_dim):
    print(i, dimx, len(lstm_dim))
    if i==0:
      # return sequences: for multi-stack, all True except last should be False
      model.add(LSTM(dimx, return_sequences=False, input_shape=(None, in_neurons), activation='tanh'))#, dropout=0.25))
    else:
      model.add(Dense(dimx, activation='tanh'))

  model.add(Dense(out_neurons, activation='linear'))
  
  model.compile(loss="mean_squared_error", optimizer=optimizer) # nadam

  return model

