# References
# https://philipperemy.github.io/keras-stateful-lstm/
#
# Copy implementation
# https://gist.github.com/Seanny123/283a8aab32330dcd3f9c2817495b4910
# which is copied in phil_lstm.py
#
# Results of testing:
# -------------------
# stateless, uniform, batch=20, data split 1000/100, epoch 15, recall len 20:
#                     loss: 0.2665 - acc: 0.9180 - val_loss: 0.2513 - val_acc: 0.9200
# ditto, batch=10:    loss: 0.4655 - acc: 0.7700 - val_loss: 0.4726 - val_acc: 0.7600 - Comment: reduced batch size => reduced accuracy
# ditto, batch=40:    loss: 0.4455 - acc: 0.8250 - val_loss: 0.3922 - val_acc: 0.8400 - Comment: increased batch size => reduced accuracy .. counter-thought
# ditto, batch=100:   loss: 0.5494 - acc: 0.7410 - val_loss: 0.5620 - val_acc: 0.7300 - Comment: further increase in batch size => further reduction in accuracy .. counter-thought
# ditto, epochs=30:   loss: 0.5600 - acc: 0.7460 - val_loss: 0.5713 - val_acc: 0.7400 - Comment: Increase epochs => slight improvement in accuracy .. expected more
# ditto, data=10k/1k: loss: 0.2263 - acc: 0.9245 - val_loss: 0.2708 - val_acc: 0.9010 - Comment: More data, larger batch size, more epochs => improvement in accuracy .. too much for so little
# ditto, batch=20:    loss: 0.6925 - acc: 0.5040 - val_loss: 0.6909 - val_acc: 0.5070 - Comment: reduced batch size => lost accuracy .. wrong direction
#                     loss: 0.3424 - acc: 0.8573 - val_loss: 0.3181 - val_acc: 0.8730 - Comment: re-run different result completely!
# ditto, epochs=15:   loss: 0.1707 - acc: 0.9517 - val_loss: 0.1787 - val_acc: 0.9470 - Comment: reduced epochs, improved accuracy .. convergence followed by divergence
# ditto, zeros:       loss: 4e-05  - acc: 1.0000 - val_loss: 3e-05  - val_acc: 1.0000 - Comment: perfection
# ditto, batch=1:     same result                                                     - Comment: perfection even though stateless
# ditto, uniform, epoch=2:
#                     loss: 0.4540 - acc: 0.8001 - val_loss: 0.3460 - val_acc: 0.8830 - Comment: convergence
# ditto, train/test 1k/100, epoch 10:
#                     loss: 0.3532 - acc: 0.8530 - val_loss: 0.3249 - val_acc: 0.8700 - Comment: also converged
# ditto, with subsampling to recall_len=10 (generated at 20): validation accuracy cannot pass 0.75
# ditto, with subsampling to recall_len=5: validation accuracy cannot pass 0.6
# ditto, zeros: same, cannot pass 0.6
# ditto, stateful: converges to accuracy = 0.85 (as without subsampling)
# ditto, batch = 10: fails to pass 0.63
# ditto, 10x more data:  still fails to pass 0.63
# ditto, batch=2: training accuracy converges very slowly to 0.73
# ditto, back to 1k/100 train/test: training accuracy converges to 0.73
# ditto, add validation in fit: validation accuracy converges to 0.65
# ditto, recal_factor=20 (like keras example on stateful lstm): validation accuracy still converges to 0.65
# ditto, batch=25: validation accuracy converges to 0.5
# ditto, train/test 10k/1k: still val acc = 0.5

recall_len = 20
batch_size = 25
epochs=15
N_train = int(10e3)
N_test = int(1e3)
recall_factor = 20

import numpy as np
from keras.models import Sequential
from keras.layers.recurrent import LSTM
from keras.models import Model
from keras.layers.core import Dense
from keras.callbacks import Callback

from numpy.random import choice
def generate_data(N=1000):
  one_indexes = choice(a=N, size=int(N / 2), replace=False)

  # X_train = np.random.uniform(-1, +1, (N, recall_len, 1))
  X_train = np.zeros((N, recall_len, 1), dtype=np.bool)

  X_train[one_indexes, 0] = 1  # very long term memory.
  
  y_train = np.zeros((N, 1))
  y_train[one_indexes] = 1
  return X_train, y_train

X_train, y_train = generate_data(N_train)
X_test, y_test   = generate_data(N_test)
print('generated data', X_train.shape, y_train.shape)

# re-stack all the data and subsample differently than construction
# PS: subsampling in the original blog post is a striding
#     Will do striding after trying the simple reshaping
def myReshape(X):
  X = X.flatten()
  new_recall = recall_len // recall_factor
  dim1 = X.shape[0] // new_recall # numeric division
  X = X.reshape((dim1, new_recall, 1))
  return X

X_train = myReshape(X_train)
X_test  = myReshape(X_test)
y_train = np.tile(y_train, recall_factor).flatten()
y_test  = np.tile(y_test,  recall_factor).flatten()
recall_len = X_train.shape[1]
print('after reshape to new recall len', X_train.shape, y_train.shape)

if False:
  print('Building STATELESS model...')
  model = Sequential()
  model.add(LSTM(10, input_shape=(recall_len, 1), return_sequences=False, stateful=False))
  # Just enabling "stateful" below is insufficient because it needs to reset state between epochs
  # https://github.com/fchollet/keras/blob/befbdaa076eedb2787ce340b1a5b4accf93d123d/examples/stateful_lstm.py
  # model.add(LSTM(10, batch_input_shape=(batch_size, recall_len, 1), return_sequences=False, stateful=True))
  model.add(Dense(1, activation='sigmoid'))
  model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
  model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs,
            validation_data=(X_test, y_test), shuffle=False)
  score, acc = model.evaluate(X_test, y_test, batch_size=batch_size, verbose=0)


print('Build STATEFUL model...')
model = Sequential()
# works
# model.add(LSTM(10, batch_input_shape=(batch_size, recall_len, 1), return_sequences=False, stateful=True))
# model.add(LSTM(10, input_shape=(recall_len, 1), return_sequences=False, stateful=False))
# like in keras example: https://github.com/fchollet/keras/blob/befbdaa076eedb2787ce340b1a5b4accf93d123d/examples/stateful_lstm.py
model.add(LSTM(10, input_shape=(recall_len, 1), batch_size=batch_size, return_sequences=False, stateful=True))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])



# print('Train...')
# for epoch in range(15):
#     mean_tr_acc = []
#     mean_tr_loss = []
# 
#     for seq_idx in range(len(X_train)):
# 
# 
# 
#         y_true = y_train[seq_idx]
# 
# 
#         for j in range(recall_len):
#             tr_loss, tr_acc = model.train_on_batch(np.expand_dims(np.expand_dims(X_train[seq_idx][j], axis=1), axis=1),
#                                                    np.array([y_true]))
#             mean_tr_acc.append(tr_acc)
#             mean_tr_loss.append(tr_loss)
#         model.reset_states()
# 
#     print('accuracy training = {}'.format(np.mean(mean_tr_acc)))
#     print('loss training = {}'.format(np.mean(mean_tr_loss)))
#     print('___________________________________')
# 
#     mean_te_acc = []
#     mean_te_loss = []
#     for seq_idx in range(len(X_test)):
#         for j in range(recall_len):
#             te_loss, te_acc = model.test_on_batch(np.expand_dims(np.expand_dims(X_test[seq_idx][j], axis=1), axis=1),
#                                                   y_test[seq_idx])
#             mean_te_acc.append(te_acc)
#             mean_te_loss.append(te_loss)
#         model.reset_states()
# 
#         for j in range(recall_len):
#             y_pred = model.predict_on_batch(np.expand_dims(np.expand_dims(X_test[seq_idx][j], axis=1), axis=1))
#         model.reset_states()
# 
#     print('accuracy testing = {}'.format(np.mean(mean_te_acc)))
#     print('loss testing = {}'.format(np.mean(mean_te_loss)))
#     print('___________________________________')


class ResetStatesCallback(Callback):
    def __init__(self):
        self.counter = 0

    def on_batch_begin(self, batch, logs={}):
        if self.counter % recall_len == 0:
            self.model.reset_states()
        self.counter += 1
        
# x = np.expand_dims(np.expand_dims(X_train.flatten(), axis=1), axis=1)
# y = np.expand_dims(np.array([[v] * recall_len for v in y_train.flatten()]).flatten(), axis=1)
x = X_train
y = y_train
model.fit(x, y, callbacks=[ResetStatesCallback()], batch_size=batch_size, shuffle=False, epochs=epochs,
          validation_data=(X_test, y_test))
score, acc = model.evaluate(X_test, y_test, batch_size=batch_size, verbose=0)

