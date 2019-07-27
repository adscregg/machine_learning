from sklearn.preprocessing import normalize
from sklearn.cluster import KMeans
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


class K_means_nD:
    """Clustering method using the KMeans algorithm.

    Parameters
    ----------
    n_clusters : int
        Number of clusters in the data.
    max_iters : int
        Maximum number of iterations before returning the result.
    tol : float
        If all the centroid centers are within tol then the loop is broken and the result returned.

    """

    def __init__(self, n_clusters=2, max_iters=100, tol=0.01):
        self.n_clusters = n_clusters
        self.max_iters = max_iters
        self.tolerance = tol

    def fit(self, k_data):
        """Fit the model to the data passed in to k_data.

        Parameters
        ----------
        k_data : Pandas DataFrame
            Pandas DataFrame with n_features, each feature being a single column.

        Returns
        -------
        self
            Returned class like object to cal methods and properties off of.

        """
        n_features = k_data.shape[1] # number of columns of the dataset
        t0 = time.time()
        k = self.n_clusters
        max_iters = self.max_iters

        centroids = self._create_centroids(self.n_clusters, k_data) # randomly create centroids

        for _ in range(max_iters):

            old_centroids = centroids # create copy of centroids variable to compare later

            cluster_num = self._calc_distances_and_assign_cluster(k_data, centroids) # distance to each cluster and assign to closest

            k_data['cluster num'] = cluster_num # create column of dataframe with cluter num

            new_centroids = self._shift_centroids(centroids, k_data) # calculate middle of clusters and shift

            centroids = new_centroids # updated centroid placement

            less_tol = self._centroid_shift_dist(centroids, old_centroids) # how much did the centers move by

            if all(less_tol): # if all centroids shifted less than the defined tolerance then break the loop
                break

        t1 = time.time()
        self.fit_runtime = t1 - t0
        self.centers = old_centroids
        self.classification = cluster_num

        return self

    def predict(self, k_data):
        """Predict the classes for the data passed into k_data.

        Parameters
        ----------
        k_data : Pandas DataFrame
            Pandas DataFrame with n_features, each feature being a single column.

        Returns
        -------
        numpy.array
            numpy array of the predicted classes.

        """
        centroids = self.centers
        classification = [] # list that will contain the classification
        for row in range(k_data.shape[0]): # for each record
            distances = [0 for i in range(len(centroids))] # list of zeros as a placeholder
            points = list(k_data.loc[row, :]) # list of each of the feature values
            for num, centroid in enumerate(centroids): # for each center
                centroid = list(centroid) # cast to a list
                distances[num] = np.sqrt(sum([(a - b) ** 2 for a, b in zip(points, centroid)])) # calculate all distances
            classification.append(np.argmin(distances)) # add index of min distance as assigned cluster
        k_data['classification'] = classification # create column with class in it
        return np.array(classification) # return array of classifications

    def fit_predict(self, X_fit, X_predict):
        """Convinience method to fit and predict data in a single method call.

        Parameters
        ----------
        X_fit : Pandas DataFrame
            Data to be fitted.
        X_predict : Pandas DataFrame
            Data to be predicted.

        Returns
        -------
        numpy.array
            numpy array of the predicted classes.

        """
        self.fit(X_fit)
        return self.predict(X_predict)

    def _create_centroids(self, n_clusters, data):
        n_features = data.shape[1] # number of columns in the dataset
        centroids = [] # list that will contain tuples of the centroid locations
        for _ in range(n_clusters): # loop depending on how many cluster have been specified
            centroid = [] # list that will contain points for single centroid
            for i in range(n_features): # for each feature
                cent_point = np.random.uniform(min(data[i]), max(data[i])) # random point within the feature max and min
                centroid.append(cent_point) # add the point to the centroid
            centroids.append(tuple(centroid)) # create tuple from list of individual points and add it to list
        return centroids

    def _calc_distances_and_assign_cluster(self, data, centroids):
        cluster_num = [None] * len(data) # no records belong to any cluster yet so they are assigned None
        for record in range(len(data)): # for row in dataset
            distances = [] # will contain distances to each centroid
            points = list(data.loc[record, :]) # feature values as a list
            for centroid in centroids: # for each centroid
                cent = list(centroid) # cast tuple to list
                dist = np.sqrt(sum([(p - c) ** 2 for p, c in zip(points, cent)])) # calculate Euclidian distance to centroid
                distances.append(dist) # add to distances list
            cluster_num[record] = np.argmin(distances) # assign to cluster than is closest

        return cluster_num

    def _shift_centroids(self, centroids, data):
        new_centroids = []
        n_features = data.shape[1] - 1 # num features, -1 as we added cluster num column which is not to be counted as a feature
        for number, centroid in enumerate(centroids): # loop through centroid and it's index
            num_k_data = data[data['cluster num'] == number] # get all data that belongs to the specific cluster
            cent_new = []
            for i in range(n_features):
                cent = np.mean(num_k_data[i]) if len(num_k_data) != 0 else np.mean(data[i]) # mean of all datapoints of feature i to get the cluster centre
                cent_new.append(cent) # add it to cent_new list
            cent_new = tuple(cent_new) # cast to a tuple
            new_centroids.append(cent_new) # new centroid tuple added to list

        return new_centroids

    def _centroid_shift_dist(self, new, old):
        centroid_shifts = []
        for index in range(len(new)):
            c0 = list(new[index]) # new position of centroid
            c1 = list(old[index]) # old position of centroid
            shift = np.sqrt(sum([(x - y)**2 for x, y in zip(c0, c1)])) # distance it has moved
            centroid_shifts.append(shift) # add shift to list
        less_tol = [elm < self.tolerance for elm in centroid_shifts] # is the shift less than defined tolerance

        return less_tol
