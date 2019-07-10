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
        return self

    def predict(self, X_test):
        pass

    def fit_predict(self, X_train, y_train, X_test):
        self.fit(X_train, y_train)
        return self.predict(X_test)
