# Embedding + LSTM for multiple features

from keras.layers import Input, Embedding, concatenate
from keras.models import Sequential
import numpy as np
from keras.layers.recurrent import LSTM
from keras.layers.core import Dense, Activation, Flatten
from keras.models import Model

#---------------------

# Data
# Input is 3D matrix: 1st dimension is time, 2nd is the sequences, 3rd is the features
#          This would be built from a 2D matrix with dimensions time x features, adding the sequences dimension
# Output is just a 1 dimensional vector
vocab_size=32
look_back = 5
n_features = 3
n_time = 1000
input_array = np.random.randint(vocab_size, size=(n_time, look_back, n_features))
target = input_array[:,-1,0].squeeze() / vocab_size

# modify input_array to become list of 2D matrices for each feature
input_array = [input_array[:,:,i].squeeze() for i in range(n_features)]

#---------------------
# Model

# input is an array of inputs, where we're using the multi-input feature of Keras
# From https://keras.io/getting-started/functional-api-guide/#multi-input-and-multi-output-models
inputs = [Input(shape=(look_back,)) for x in range(n_features)]

# Choose the desired dimensionality of the embedding
desired_dimension = 2

# proceed with embedding each feature matrix
x = [Embedding(input_dim=vocab_size, output_dim=desired_dimension, input_length=look_back)(y) for y in inputs]

# concatenate all embedding outputs
x = concatenate(x)

# regular keras-fu from here on
x = LSTM(5)(x)
output = Dense(1, activation='sigmoid')(x)
model = Model(inputs=inputs, outputs=[output]) # <<< set multi-input to single-output here
model.compile(loss='binary_crossentropy', optimizer='rmsprop')
model.summary()

#-------------------
# Training
model.fit(
  [x[:(n_time*2//3),:] for x in input_array],
  y=target[:(n_time*2//3)],
  epochs=125, verbose=0, validation_split=0.2, shuffle=True
)

#---------------------
# Prediction for the 1st time point for example
# Note that "0:1" below will output the 0 index, while maintaining the dimension
# Also note that the passed argument is a lits of matrices
print("test score", model.evaluate([x[:(n_time*2//3),:] for x in input_array], target[:(n_time*2//3)]))

"""
predicted = model.predict([x[:(n_time*2//3),:] for x in input_array])

from matplotlib import pyplot as plt
plt.plot(predicted)
plt.plot(target[:(n_time*2//3)])
plt.title("prediction vs actual")
plt.show()
"""
