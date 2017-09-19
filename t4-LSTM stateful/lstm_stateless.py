# https://github.com/fchollet/keras/blob/befbdaa076eedb2787ce340b1a5b4accf93d123d/examples/stateful_lstm.py

'''Example script showing how to use stateful RNNs
to model long sequences efficiently.
'''
from __future__ import print_function
import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense, LSTM


# since we are using stateless rnn,
# tsteps needs to be >= lahead for the training to converge
tsteps = 2

# choosing a smaller batch size would be slower, but converges earlier (epoch-wise)
batch_size = 100
epochs = 25
# number of elements ahead that are used to make the prediction
lahead = tsteps


def gen_cosine_amp(amp=100, period=1000, x0=0, xn=50000, step=1, k=0.0001):
    """Generates an absolute cosine time series with the amplitude
    exponentially decreasing

    Arguments:
        amp: amplitude of the cosine function
        period: period of the cosine function
        x0: initial x of the time series
        xn: final x of the time series
        step: step of the time series discretization
        k: exponential rate
    """
    cos = np.zeros(((xn - x0) * step, 1, 1))
    for i in range(len(cos)):
        idx = x0 + i * step
        cos[i, 0, 0] = amp * np.cos(2 * np.pi * idx / period)
        cos[i, 0, 0] = cos[i, 0, 0] * np.exp(-k * idx)
    return cos


print('Generating Data...')
# cos = gen_cosine_amp()
cos = np.random.uniform(-100, +100, (int(50e3)//lahead, lahead, 1))
print('Input shape:', cos.shape)

expected_output = np.mean(cos, axis=1)

print('Output shape:', expected_output.shape)

# print('Plotting input/output')
# plt.subplot(2, 1, 1)
# plt.plot(cos.flatten())
# plt.title('Input')
# plt.subplot(2, 1, 2)
# plt.plot(expected_output)
# plt.title('Expected output')
# plt.show()
# 
# print('Plotting input - output')
# plt.plot(cos.flatten() - expected_output.flatten())
# plt.title('input - output is negligible, but non-zero')
# plt.show()

print('Creating Model...')
model = Sequential()
model.add(LSTM(50,
               input_shape=(tsteps, 1),
               batch_size=batch_size,
               return_sequences=True,
               stateful=True))
model.add(LSTM(50,
               return_sequences=False,
               stateful=True))
model.add(Dense(1))
model.compile(loss='mse', optimizer='rmsprop')

print('Training')
model.fit(cos, expected_output,
      batch_size=batch_size,
      epochs=epochs,
      verbose=1,
      shuffle=False)

print('Evaluating on new data')
cos = np.random.uniform(-100, +100, (int(50e3)//lahead, lahead, 1))
expected_output = np.mean(cos, axis=1)
print(cos.shape, expected_output.shape, batch_size)
score = model.evaluate(cos, expected_output, batch_size=batch_size, verbose=2)
print('score', score)

# print('Predicting')
# predicted_output = model.predict(cos, batch_size=batch_size)
# 
# print('Plotting Results')
# plt.subplot(2, 1, 1)
# plt.plot(expected_output)
# plt.title('Expected')
# plt.subplot(2, 1, 2)
# plt.plot(predicted_output)
# plt.title('Predicted')
# plt.show()
