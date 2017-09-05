# Deps
# pip install unittest-data-provider
#
# Run all
# python -m unittest -q test_p1_core
#
# Run a single test class with unittest
# http://pythontesting.net/framework/specify-test-unittest-nosetests-pytest/
# python -m unittest -q test_p1_core.P1CoreCase.test_fit

import unittest
from unittest_data_provider import data_provider
import p1_core
import utils2

class P1CoreCase(unittest.TestCase):
  # https://stackoverflow.com/questions/18905637/python-unittest-data-provider#18906125
  params = lambda: (
    # passed (int(10e3), [30],  300),
    # passed (int( 1e3), [30], 3000),
    # passed (int( 1e3), [60], 2100),
    # passed (int( 1e3), [30, 20, 10], 4000),
    (int(10e3), [90,60,30],  300),
  )

  @data_provider(params)
  def test_fit(self, nb_samples, lstm_dim, epochs):
    (X_model, Y, lags) = p1_core.data(nb_samples)
    # model = p1_core.model(X_model.shape[1], lstm_dim)

    look_back = 5
    model = utils2.build_lstm_ae(X_model.shape[1], lstm_dim[0], look_back, lstm_dim[1:], "adam", 1)
    # model.summary()
    (history, err) = p1_core.fit(X_model, Y, lags, model, epochs, look_back)
    self.assertLess(err, 0.01)
