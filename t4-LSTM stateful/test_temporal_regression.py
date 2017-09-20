# copied from keras/tests/integration_tests/test_temporal_data_tasks.py#test_temporal_regression

import numpy as np
from keras.utils.test_utils import get_test_data
from keras.models import Sequential
from keras import layers
import keras

import timeit

def doit(epochs,batch_size,layer_type):
	start_time = timeit.default_timer()

#	np.random.seed(1337)
	(x_train, y_train), (x_test, y_test) = get_test_data(num_train=2000,
							 num_test=200,
							 input_shape=(5,) if layer_type=='Dense' else (3, 5),
							 output_shape=(2,),
							 classification=False)

	model = Sequential()
	if layer_type=='Dense':
		model.add(layers.Dense(y_train.shape[-1],
				 input_shape=(x_train.shape[1],)))
	else:
		model.add(layers.LSTM(y_train.shape[-1],
				  input_shape=(x_train.shape[1], x_train.shape[2])))
	model.compile(loss='mse', optimizer='adam') # hinge
	history = model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size,
			validation_data=(x_test, y_test), verbose=0)
	print(
		'epochs %-5s, batch size %-5s, layer type %-10s: final loss %0.2f, seconds %0.2f'
		%(
		epochs,      batch_size,     layer_type,     history.history['loss'][-1], timeit.default_timer() - start_time
		)
	)

"""
	np.random.seed(1338)
	(x_train, y_train), (x_test, y_test) = get_test_data(num_train=2000,
							 num_test=200,
							 input_shape=(5,) if layer_type=='Dense' else (3, 5),
							 output_shape=(2,),
							 classification=False)
	score_train = model.evaluate(x_train, y_train, verbose=0)
	score_test = model.evaluate(x_test, y_test, verbose=0)
	score_train_1 = model.evaluate(x_train[:1000 ], y_train[:1000 ], verbose=0)
	score_train_2 = model.evaluate(x_train[ 1000:], y_train[ 1000:], verbose=0)

	print('%0.2f, %0.2f, %0.2f, %0.2f' % (score_train, score_test, score_train_1, score_train_2))
"""

doit( 15, 16,'Dense')
doit( 15,160,'Dense')
doit(150,160,'Dense')

"""
doit( 15, 20,'Dense')
doit( 15,200,'Dense')
doit(150,200,'Dense')

doit( 15, 20,'Lstm')
doit( 15,200,'Lstm')
doit(150,200,'Lstm')
"""
