# Deps
# pip install unittest-data-provider
#
# Run all
# python -m unittest -q test_p1_core
#
# Run a single test class with unittest
# http://pythontesting.net/framework/specify-test-unittest-nosetests-pytest/
# python -m unittest -q test_p1_core.P1CoreCase.test_fit_model_1
# python -m unittest -q test_p1_core.P1CoreCase.test_fit_model_2


import unittest
from unittest_data_provider import data_provider
import p1_core
import utils2

import numpy as np
import pandas as pd

class P1CoreCase(unittest.TestCase):

  # simulated data (copy from p5g)
  # nb_samples = int(1e3)
  def data(self,nb_samples:int):
    if nb_samples<=0: raise Exception("nb_samples <= 0")

    lags = [1, 2]
    X1 = pd.Series(np.random.randn(nb_samples))
    X2 = pd.Series(np.random.randn(nb_samples))
    # https://stackoverflow.com/a/20410720/4126114
    X_model = pd.concat({'main': X1, 'lagged 1': X1.shift(lags[0]), 'lagged 2': X1.shift(lags[1]), 'new': X2}, axis=1).dropna()
                         
    X_model['mult'] = X_model.apply(lambda row: row[2]*row[3], axis=1)
    
    
    # Y = X_model.apply(lambda row: 0.25*row[0] + 0.25*row[1] + 0.25*row[2] + 0.25*row[3], axis=1)
    
    Y = X_model.apply(lambda row: 0.2*row['main'] + 0.2*row['lagged 1'] + 0.2*row['lagged 2'] + 0.2*row['new'] + 0.2*row['mult'], axis=1)
    Y = Y.values.reshape((Y.shape[0],1))

    # drop columns in X_model that LSTM is supposed to figure out
    del X_model['lagged 1']
    del X_model['lagged 2']
    del X_model['mult']

    return (X_model, Y, lags)

  #-------------------------
  # https://stackoverflow.com/questions/18905637/python-unittest-data-provider#18906125
  params = lambda: (
    # (int(10e3),  600, 0.0196, [10]),
    # (int(10e3),  600, 0.0147, [10,10]),
    # (int(10e3),  600, 0.0173, [10,10,10]),
    # (int(10e3),  600, 0.0528, [10,10,10,10]),
    # (int(10e3),  600, 0.0093, [30]),
    # (int(10e3),  600, 0.0097, [60]),
    # (int(10e3),  600, 0.0061, [90]),
    # (int(10e3),  600, 0.0146, [30,10]),
    # (int(10e3),  600, 0.0082, [30,30]),
    # (int(10e3),  600, 0.0085, [30,60]),
    # (int(10e3),  600, 0.0079, [60,30]),
    # (int(10e3),  600, 0.0192, [60,60]),
    # (int(10e3),  600, 0.0054, [90,60]),
    # (int(10e3),  600, 0.0086, [90,60,30]),

    # tests with less epochs
    # (int(10e3),  400, 0.01, [90,60,30]),
    # (int(10e3),  400, 0.01, [30,30]),
    # (int(10e3),  300, 0.0129, [60,30]),

    # failed tests
    # stuck since epoch 400 # (int(10e3), 1000, 0.01, [30,20,10]),

    # tests with less data
    # (int( 1e3), 3000, 0.01, [30]),
    # (int( 1e3), 2100, 0.01, [60]),
    # (int( 1e3), 4000, 0.01, [30, 20, 10]),
  )

  #-------------------------
  @data_provider(params)
  def test_fit_model_1(self, nb_samples, epochs, expected_mse, lstm_dim):
    (X_model, Y, lags) = self.data(nb_samples)

    look_back = 5
    model = p1_core.model(X_model.shape[1], lstm_dim, look_back)
    # model = utils2.build_lstm_ae(X_model.shape[1], lstm_dim[0], look_back, lstm_dim[1:], "adam", 1)
    model.summary()
    (history, err) = p1_core.fit(X_model, Y, lags, model, epochs, look_back)

    # with 10e3 points
    #      np.linalg.norm of data = 45
    #      and a desired mse <= 0.01
    # The minimum loss required = (45 * 0.01)**2 / 10e3 ~ 2e-5
    #
    # with 1e3 points
    #      np.linalg.norm of data = 14
    #      and a desired mse <= 0.01
    # The minimum loss required = (14 * 0.01)**2 / 1e3 ~ 2e-5 (also)
    self.assertLess(err, expected_mse)

  #-------------------------
  @data_provider(params)
  def test_fit_model_2(self, nb_samples, epochs, expected_mse, lstm_dim):
    (X_model, Y, lags) = self.data(nb_samples)

    look_back = 5
    model = p1_core.model_2(X_model.shape[1], lstm_dim, look_back)
    # model = utils2.build_lstm_ae(X_model.shape[1], lstm_dim[0], look_back, lstm_dim[1:], "adam", 1)
    model.summary()
    (history, err) = p1_core.fit(X_model, Y, lags, model, epochs, look_back)

    # with 10e3 points
    #      np.linalg.norm of data = 45
    #      and a desired mse <= 0.01
    # The minimum loss required = (45 * 0.01)**2 / 10e3 ~ 2e-5
    #
    # with 1e3 points
    #      np.linalg.norm of data = 14
    #      and a desired mse <= 0.01
    # The minimum loss required = (14 * 0.01)**2 / 1e3 ~ 2e-5 (also)
    self.assertLess(err, expected_mse)
