# libraries (copy from p5g)

# import os
import numpy as np
from keras.models import Sequential
from keras.layers.core import Dense, Dropout
from keras.layers.recurrent import SimpleRNN, LSTM, GRU
import keras

import pandas as pd

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

