from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import normalize
from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns

import numpy as np
import pandas as pd
import random

import time
import os

import warnings
import random_data


class Logistic_Regression_Binary:
    def __init__(self, learn_rate = 0.01, iterations = 200):
        self.learn_rate = learn_rate
        self.iters = iterations


    def fit(self, X_log_fit, y_log_fit):
        n_feats = X_log_fit.shape[1]
        m = X_log_fit.shape[0]
        self.m = m
        weights = np.zeros((1,n_feats))
        bias = 0
        cost = np.inf

        for i in range(self.iters):
            if i % 20 == 0:
                print('After {num} iterations the cost is {c}'.format(num = i, c = cost))

            y_T = np.transpose(y_log_fit)
            prev_cost = cost

            lin_eqn = np.dot(weights, np.transpose(X_log_fit)) + bias
            result = self._sigmoid(lin_eqn)


            cost = self._cost_function(result, y_T)
            dw, db = self._shift_weights_and_bias_by(X_log_fit, y_log_fit, result)
            weights, bias = self._update_weights_and_bias(weights, bias, dw, db)

            if prev_cost - cost < 0.001:
                print('After {num} iterations the cost is {c}'.format(num = i, c = cost))
                break

        self.w = weights
        self.b = bias

        return self

    def _sigmoid(self, lin_eqn):
        return 1/(1 + np.exp(-lin_eqn))

    def _cost_function(self, result, targets_T):
        m = self.m
        return (-1/m)*(np.sum((targets_T*np.log(result)) + ((1 - targets_T)*np.log(1-result + 0.000001))))

    def _update_weights_and_bias(self, weights, bias, dw, db):
        w = weights - (self.learn_rate * np.transpose(dw))
        b = bias - (self.learn_rate * db)
        return w, b

    def _shift_weights_and_bias_by(self, data, target, guess):
        m = self.m
        dw = (1/m)*(np.dot(np.transpose(data), np.transpose(guess-np.transpose(target))))
        db = (1/m)*(np.sum(guess-np.transpose(target)))
        return dw, db


    def predict(self, log_predict):
        pred_val = np.dot(self.w, np.transpose(log_predict)) + self.b
        pred = 1/(1 + np.exp(-pred_val))
        prediction = np.array([1 if p > 0.5 else 0 for p in pred[0]])

        return prediction

    def fit_predict(self, X_log_fit, y_log_fit, log_predict):
        self.fit(X_log_fit, y_log_fit)
        return self.predict(log_predict)
