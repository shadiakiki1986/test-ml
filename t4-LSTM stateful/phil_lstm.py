# copy from https://gist.github.com/Seanny123/283a8aab32330dcd3f9c2817495b4910#file-phil_lstm-py-L9
# which in its turns is based on https://philipperemy.github.io/keras-stateful-lstm/
#
# Related: test_philippe.py

from keras.models import Sequential
from keras.layers import LSTM, Dense
import numpy as np


def gen_sig(num_samples, seq_len):
    one_indices = np.random.choice(a=num_samples, size=num_samples // 2, replace=False)

    x_val = np.zeros((num_samples, seq_len), dtype=np.bool)
    x_val[one_indices, 0] = 1

    y_val = np.zeros(num_samples, dtype=np.bool)
    y_val[one_indices] = 1

    return x_val, y_val


N_train = 100
N_test = 10
recall_len = 20

X_train, y_train = gen_sig(N_train, recall_len)

X_test, y_test = gen_sig(N_train, recall_len)

print(X_train.shape, y_train.shape)

print('Build STATEFUL model...')
model = Sequential()
model.add(LSTM(10, batch_input_shape=(1, 1, 1), return_sequences=False, stateful=True))
# usually you use softmax for classification, but that doesn't work because
# the output is 1 dimensional
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

print('Train...')
for epoch in range(15):
    mean_tr_acc = []
    mean_tr_loss = []

    for seq_idx in range(X_train.shape[0]):
        start_val = X_train[seq_idx, 0]
        assert y_train[seq_idx] == start_val
        assert tuple(np.nonzero(X_train[seq_idx, :]))[0].shape[0] == start_val

        y_in = np.array([y_train[seq_idx]], dtype=np.bool)

        for j in range(recall_len):
            x_in = np.array([[[X_train[seq_idx][j]]]])
            tr_loss, tr_acc = model.train_on_batch(x_in, y_in)

            mean_tr_acc.append(tr_acc)
            mean_tr_loss.append(tr_loss)

        # move this into the inner loop and watch the network never learn
        # move into inner loop and reset every x < recall_len for fancy failure
        model.reset_states()

    print('accuracy training = {}'.format(np.mean(mean_tr_acc)))
    print('loss training = {}'.format(np.mean(mean_tr_loss)))
    print('___________________________________')

    mean_te_acc = []
    mean_te_loss = []
    for seq_idx in range(X_test.shape[0]):
        start_val = X_test[seq_idx, 0]
        assert y_test[seq_idx] == start_val
        assert tuple(np.nonzero(X_test[seq_idx, :]))[0].shape[0] == start_val

        y_in = np.array([y_test[seq_idx]], dtype=np.bool)

        for j in range(recall_len):
            te_loss, te_acc = model.test_on_batch(np.array([[[X_test[seq_idx][j]]]], dtype=np.bool), y_in)
            mean_te_acc.append(te_acc)
            mean_te_loss.append(te_loss)
        model.reset_states()

    print('accuracy testing = {}'.format(np.mean(mean_te_acc)))
    print('loss testing = {}'.format(np.mean(mean_te_loss)))
    print('___________________________________')
