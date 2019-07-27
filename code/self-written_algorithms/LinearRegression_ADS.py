from sklearn.linear_model import LinearRegression
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




class Linear_Regression:
    def __init__(self):
        pass

    def fit(self, fit_data, fit_data_target):
        fit_data = np.array(fit_data) # make sure data is numpy array
        inputs = np.c_[fit_data, np.ones(fit_data.shape[0])] # add column of ones to end of dataset
        outputs = np.array(fit_data_target) # array of targets
        in_T = np.transpose(inputs) # transpose of the input data

        coef_matrix = in_T @ inputs # matrix multiplication to give square matrix representing coeficients in linear system of equations
        results = in_T @ outs # mat mul to give a column vector of outputs of system of equations

        self.lin_coefs = np.linalg.solve(coef_matrix, results) # solve system of equations

        self.weights = self.lin_coefs[:-1] # weights
        self.intercept = self.lin_coefs[-1] # intercept

        return self

    def predict(self, X_test):
        X_test = np.array(X_test).reshape((X_test.shape[0],-1)) # reshape into a 2D array
        predictions = X_test @ self.weights + self.intercept # run vectorized linear model
        return predictions

    def fit_predict(self, X_train, y_train, X_test):
        self.fit(X_train, y_train)
        return self.predict(X_test)
