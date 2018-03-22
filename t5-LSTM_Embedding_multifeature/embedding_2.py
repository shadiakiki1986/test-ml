from keras.layers import Embedding
from keras.models import Sequential
import numpy as np
from keras.layers.recurrent import LSTM
from keras.layers.core import Dense, Activation, Flatten

input_array = np.random.randint(1000, size=(32, 10))
target = input_array[:,0].squeeze()

model = Sequential()

vocab_size = 1000
model.add(Embedding(vocab_size, 10))
model.add(LSTM(5))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='rmsprop')
model.fit(input_array, y=target,  epochs=5, verbose=2, validation_split=0.2, shuffle=True)

print(model.predict(input_array[0,:]))
