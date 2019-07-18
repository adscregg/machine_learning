import pandas as pd
import numpy as np
import random
from scipy import stats

import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('darkgrid')

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.feature_selection import SelectKBest
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, mean_squared_error

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import MultinomialNB

wine = pd.read_csv('../../datasets/winequalityN.csv')

print(wine.shape, '\n\n')

print(wine.head(), '\n\n')

print(wine.describe(), '\n\n')

print(wine.info(), '\n\n')

print(wine.isna().sum(), '\n\n')


class impute_from_distribution(BaseEstimator, TransformerMixin):
    def __init__(self, col):
        self.col = col
        self.target_col = 'type'
    def fit(self, X, y = None):

        self.white_dist, _ = np.histogram(X[(X[self.target_col] == 'white')][self.col].dropna(), bins = 75)
        self.red_dist, _ = np.histogram(X[(X[self.target_col] == 'red')][self.col].dropna(), bins = 75)

        X[self.col] = X[[self.col, self.target_col]].apply(self._impute, axis = 1)

        return X

    def transform(self, X, y = None):
        return X

    def fit_transform(self, X, y = None):
        X = self.fit(X)
        return self.transform(X)

    def _impute(self, data):
        value = data[0]
        type = data[1]
        if np.isnan(value):
            if type == 'white':
                return np.quantile(self.white_dist, random.uniform(0,1))
            else:
                return np.quantile(self.red_dist, random.uniform(0,1))
        else:
            return value

def wine_preprocessing(data):

    wine_pipeline = Pipeline([('fixed_acidity', impute_from_distribution('fixed acidity')),
                                ('volatile_acidity', impute_from_distribution('volatile acidity')),
                                ('citric_acid', impute_from_distribution('citric acid')),
                                ('residual_sugar', impute_from_distribution('residual sugar')),
                                ('chlorides', impute_from_distribution('chlorides')),
                                ('pH', impute_from_distribution('pH')),
                                ('sulphates', impute_from_distribution('sulphates'))])

    wine = wine_pipeline.fit_transform(data)

    data['type'] = data['type'].map({'white': 1, 'red': 0})

    scaler = MinMaxScaler()
    wine_scaled = scaler.fit_transform(data.drop('type', axis = 1), data['type'])

    feat_cols = list(data.columns)
    feat_cols.remove('type')
    data[feat_cols] = wine_scaled

    return data

wine = wine_preprocessing(wine)

print(wine.head())

X_train, X_test, y_train, y_test = train_test_split(wine.drop('type', axis = 1), wine['type'], test_size=0.33)

svc = SVC(gamma='scale')
log = LogisticRegression(solver='lbfgs')
rf = RandomForestClassifier()
KNN = KNeighborsClassifier()

svc_gs = GridSearchCV(svc, {'C':[0.01,0.1,1,10,30,50,100], 'kernel':['rbf', 'sigmoid']}, cv=5)
log_gs = GridSearchCV(log, {'C':[0.01,0.1,1,10,30,50,100]}, cv=5)
rf_gs = GridSearchCV(rf, {'n_estimators':[5, 10, 20,30,50,100]}, cv=5)
KNN_gs = GridSearchCV(KNN, {'n_neighbors':[3,5,7,10,15,25,30,40]}, cv=5)

svc_pred = svc_gs.fit(X_train, y_train).predict(X_test)
log_pred = log_gs.fit(X_train, y_train).predict(X_test)
rf_pred = rf_gs.fit(X_train, y_train).predict(X_test)
KNN_pred = KNN_gs.fit(X_train, y_train).predict(X_test)

print('SVC acc: ', accuracy_score(y_test,svc_pred))
print('Log acc: ', accuracy_score(y_test, log_pred))
print('RF acc: ', accuracy_score(y_test, rf_pred))
print('KNN acc: ', accuracy_score(y_test, KNN_pred))

print('SVC')
print(classification_report(y_test,svc_pred)) #SVC
print(confusion_matrix(y_test,svc_pred))

print('Log')
print(classification_report(y_test, log_pred)) #LogisticRegression
print(confusion_matrix(y_test, log_pred))

print('RF')
print(classification_report(y_test, rf_pred)) #RandomForest
print(confusion_matrix(y_test, rf_pred))

print('KNN')
print(classification_report(y_test, KNN_pred)) #KNN
print(confusion_matrix(y_test, KNN_pred))
