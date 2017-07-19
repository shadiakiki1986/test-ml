# https://gist.github.com/shadiakiki1986/2c293e364563492c65bffdb6122b4e92
from sklearn.preprocessing import MinMaxScaler #  normalize,
min_max_scaler = MinMaxScaler()
# def myNorm3(X): return normalize(X, norm='l2', axis=0)
def myNorm3(X): return min_max_scaler.fit_transform(X)

##########################################
import numpy as np
from matplotlib import pyplot as plt
def myPlot(X):
    X_plt = X+5*np.arange(X.shape[1])
    N_PLOT=200
    plt.plot(X_plt[0:N_PLOT,:])
    plt.show()

##############################################
# could make wrapper from https://gist.github.com/ktrnka/81c8a7b79cb05c577aab
# and make pipeline
# copied from simple example at https://blog.keras.io/building-autoencoders-in-keras.html
from keras.layers import Input, Dense, Dropout
from keras.models import Model, load_model
from keras.layers.advanced_activations import LeakyReLU #, PReLU

def buildNetwork(input_shape:int):
    input_img = Input(shape=(input_shape,))
    encoded = input_img
    encoding_dim_ae = 2
    # encoded = Dense( encoding_dim_ae, activation='relu' )(encoded)

    # hidden layer
    encoded = Dense( encoding_dim_ae, activation='linear' )(encoded)

    # use leaky relu
    # https://github.com/fchollet/keras/issues/117
    encoded = LeakyReLU(alpha=.3)(encoded)   # add an advanced activation

    # GET DEEP
    # encoding_dim2 = 50
    # encoding_dim3 = 10
    # encoded = Dense(encoding_dim2, activation='relu')(encoded)
    # encoded = Dense(encoding_dim3, activation='relu')(encoded)
    # decoded = Dense(encoding_dim2, activation='relu')(encoded)

    decoded = Dense(input_shape, activation='sigmoid')(encoded)

    autoencoder = Model(input_img, decoded)
    encoder = Model(input_img, encoded)

    # encoded_input = Input(shape=(encoding_dim_ae,))
    # decoder_layer = autoencoder.layers[-1]
    # decoder = Model(encoded_input, decoder_layer(encoded_input))

    # other: optimizer='adadelta', loss='binary_crossentropy'
    autoencoder.compile(optimizer='rmsprop', loss='mean_squared_error')
    encoder.compile(optimizer='rmsprop', loss='mean_squared_error')
    # decoder.compile(optimizer='rmsprop', loss='mean_squared_error')
    
    return (autoencoder, encoder)
