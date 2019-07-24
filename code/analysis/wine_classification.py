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
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, mean_squared_error

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import MultinomialNB

import warnings

warnings.filterwarnings('ignore')




wine = pd.read_csv('../../datasets/winequalityN.csv') #read in the dataset

(wine.head(), '\n\n') #view the first 5 rows

(wine.describe(), '\n\n') #see the summary statistics of the numerical data

(wine.info(), '\n\n') #get info on what type of data is in each column as well as how many non null values each one contains

(wine.isna().sum(), '\n\n') # how many null values does each column contain


plt.figure(figsize=(12,8))
sns.countplot(wine['type']) # count the number of each class
plt.show()


plt.figure(figsize=(12,8))
sns.boxplot(wine['fixed acidity']) #boxplot of the values of the 'fixed acidity' column
plt.show()


plt.figure(figsize=(12,8))
sns.distplot(wine['citric acid'].dropna(), rug = True) # plot the distribution as well as rugs for each data point
plt.xlim(0) # set the limit of the x axis to start at 0
plt.show()


plt.figure(figsize=(12,8))
sns.heatmap(wine.corr(), vmax = 1, vmin = -1, cmap = 'coolwarm') # heatmap
plt.show()


sns.violinplot(wine['residual sugar']) # violin plot, similar to boxplot but also shows density
plt.show()


class impute_from_distribution(BaseEstimator, TransformerMixin): # inherit from sklearn.base classes to create own pipeline function
    def __init__(self, col):
        self.col = col
        self.target_col = 'type' # automatically set the target column to 'target'
    def fit(self, X, y = None):

        _ , self.white_dist = np.histogram(X[(X[self.target_col] == 'white')][self.col].dropna(), bins = 75) # data into 75 bins
        _ , self.red_dist = np.histogram(X[(X[self.target_col] == 'red')][self.col].dropna(), bins = 75) # data into 75 bins

        X[self.col] = X[[self.col, self.target_col]].apply(self._impute, axis = 1) # impute the missing values

        return X

    def transform(self, X, y = None): # transform data
        return X

    def fit_transform(self, X, y = None): # run fit and transform
        X = self.fit(X)
        return self.transform(X)

    def _impute(self, data):
        value = data[0] # the value to be imputed
        type_ = data[1] # the class
        if np.isnan(value): # if the value is NaN
            if type_ == 'white': # if class is 'white'
                return np.quantile(self.white_dist, random.uniform(0,1)) # inverse CDF sampling
            else:
                return np.quantile(self.red_dist, random.uniform(0,1)) # inverse CDF sampling
        else:
            return value # if not NaN return the sample value




def wine_preprocessing(data): # create a pipline function for the wine dataset

    wine_pipeline = Pipeline([('fixed_acidity', impute_from_distribution('fixed acidity')), # run custom function on each column
                                ('volatile_acidity', impute_from_distribution('volatile acidity')), # containing a missing value
                                ('citric_acid', impute_from_distribution('citric acid')),
                                ('residual_sugar', impute_from_distribution('residual sugar')),
                                ('chlorides', impute_from_distribution('chlorides')),
                                ('pH', impute_from_distribution('pH')),
                                ('sulphates', impute_from_distribution('sulphates'))])

    wine = wine_pipeline.fit_transform(data) # fit and transform the data

    data['type'] = data['type'].map({'white': 1, 'red': 0}) # make the classes numerical so it can be handled by sklearn algorithms

    scaler = MinMaxScaler() # scale the data
    wine_scaled = scaler.fit_transform(data.drop('type', axis = 1), data['type']) # fit and transform the data by the fitted scalar

    feat_cols = list(data.columns)
    feat_cols.remove('type')
    data[feat_cols] = wine_scaled # redefine with scaled columns

    return data



wine = wine_preprocessing(wine) # process the dataset
wine.head() # view the first 5 rows of the new dataset



X_train, X_test, y_train, y_test = train_test_split(wine.drop('type', axis = 1), wine['type'], test_size=0.33, random_state = 42) # split the dataset



svc = SVC(gamma='scale') # define each of the classifiers
log = LogisticRegression(solver='lbfgs')
rf = RandomForestClassifier()
KNN = KNeighborsClassifier()


print('\nWe are now running GridSearchCV to find the best hyperparameters for each of our models to ensure that they perform the best that they can with the options we have given them, especially as we have a large class imbalance.\n')


svc_gs = GridSearchCV(svc, {'C':[0.01,0.1,1,10,30,50,100], 'kernel':['rbf', 'sigmoid']}, cv=5) # define hyper parameters to be tested to maximise results
log_gs = GridSearchCV(log, {'C':[0.01,0.1,1,10,30,50,100]}, cv=5)
rf_gs = GridSearchCV(rf, {'n_estimators':[5, 10, 20,30,50,100]}, cv=5)
KNN_gs = GridSearchCV(KNN, {'n_neighbors':[3,5,7,10,15,25,30,40]}, cv=5)


print('Fitting models and predicting values...')

svc_pred = svc_gs.fit(X_train, y_train).predict(X_test) # fit the models and predict off of them
log_pred = log_gs.fit(X_train, y_train).predict(X_test)
rf_pred = rf_gs.fit(X_train, y_train).predict(X_test)
KNN_pred = KNN_gs.fit(X_train, y_train).predict(X_test)


print('Accuracy Scores: \n')
print('SVC acc: ', accuracy_score(y_test,svc_pred)) # see the accuracy scores
print('Log acc: ', accuracy_score(y_test, log_pred))
print('RF acc: ', accuracy_score(y_test, rf_pred))
print('KNN acc: ', accuracy_score(y_test, KNN_pred))


print('Although these results look promising, do not forget that we had a very significant class imbalance which may have affected our models. Lets look at the classification report and confusion matrix for each of the models.')

print('\n SVC')
print(classification_report(y_test,svc_pred)) #SVC
print(confusion_matrix(y_test,svc_pred)) # see the classification report and confusion matrix


print('\n Log')
print(classification_report(y_test, log_pred)) #LogisticRegression
print(confusion_matrix(y_test, log_pred))


print('\n RF')
print(classification_report(y_test, rf_pred)) #RandomForest
print(confusion_matrix(y_test, rf_pred))


print('\n KNN')
print(classification_report(y_test, KNN_pred)) #KNN
print(confusion_matrix(y_test, KNN_pred))


print('All models still appear to be very good at predicting both classes but there is still some imbalance in the scores, it is noticible that class 0 (red) always has slightly lower scores in the classification report which would indicate that the dataset has had some influence on each of the models. However, in this case the models still perform well even with the different metrics in the classification report.\n')




print('Best Log estimator: ' + str(log_gs.best_estimator_), '\n\n') # see what parameters the best estimator had from running GridSearchCV


print('Best SVC estimator: ' + str(svc_gs.best_estimator_), '\n\n')

print('\nThe two most accurate classifiers have a large C value, this means that they penalize wrong classifications more during the training process and hence are \'persuaded\' not to always predict the larger class. Below is an example of fitting our models without tuning the hyper parameters.\n')



l = LogisticRegression().fit(X_train, y_train)
p = l.predict(X_test)
print('Log\n')
print(classification_report(y_test, p))
print(confusion_matrix(y_test, p))
print(accuracy_score(y_test, p))



s = SVC().fit(X_train, y_train)
p_1 = s.predict(X_test)
print('SVC')
print(classification_report(y_test, p_1))
print(confusion_matrix(y_test, p_1))
print(accuracy_score(y_test, p_1))


print('It is more noticable that the class difference is having an affect on the overall performance of the models. Note not only the drop in accuracy as a measure of performance but the difference of the the values of the metrics between class 0 and class 1, this clearly demonstrates that the imbalance is having an impact, albeit not significant in this case as the models are still able to accurately predict the class, however, this effect is still one to note when creating models as it may be more significant with other datasets.')
