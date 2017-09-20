# copied from keras/tests/integration_tests/test_temporal_data_tasks.py#test_temporal_regression

import numpy as np
from keras.utils.test_utils import get_test_data
from keras.models import Sequential
from keras import layers
import keras

import timeit

def doit(epochs,batch_size,layer_type):
	start_time = timeit.default_timer()

	np.random.seed(1337)
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
	model.compile(loss='hinge', optimizer='adam')
	history = model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size,
			validation_data=(x_test, y_test), verbose=0)
	print(
		'epochs %s, batch size %s, layer type %s: final loss %s,              seconds %s'
		%(
		epochs,      batch_size,     layer_type,     history.history['loss'][-1], timeit.default_timer() - start_time
		)
	)


doit( 15, 16,'Dense')
doit(150,160,'Dense')
doit( 15,160,'Dense')

doit( 15, 16,'Lstm')
doit(150,160,'Lstm')
doit( 15,160,'Lstm')

