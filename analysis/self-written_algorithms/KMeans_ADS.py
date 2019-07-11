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
        n_features = k_data.shape[1]
        t0 = time.time()
        k = self.n_clusters
        max_iters = self.max_iters

        centroids = self._create_centroids(self.n_clusters, k_data)

        for _ in range(max_iters):

            old_centroids = centroids

            cluster_num = self._calc_distances_and_assign_cluster(k_data, centroids)

            k_data['cluster num'] = cluster_num

            new_centroids = self._shift_centroids(centroids, k_data)

            centroids = new_centroids

            less_tol = self._centroid_shift_dist(centroids, old_centroids)

            if all(less_tol):
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
        classification = []
        for row in range(k_data.shape[0]):
            distances = [0 for i in range(len(centroids))]
            points = list(k_data.loc[row, :])
            for num, centroid in enumerate(centroids):
                centroid = list(centroid)
                distances[num] = np.sqrt(sum([(a - b) ** 2 for a, b in zip(points, centroid)]))
            classification.append(np.argmin(distances))
        k_data['classification'] = classification
        return np.array(classification)

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
        n_features = data.shape[1]
        centroids = []
        for _ in range(n_clusters):
            centroid = []
            for i in range(n_features):
                cent_point = np.random.uniform(min(data[i]), max(data[i]))
                centroid.append(cent_point)
            centroids.append(tuple(centroid))
        return centroids

    def _calc_distances_and_assign_cluster(self, data, centroids):
        cluster_num = [None] * len(data)
        for record in range(len(data)):
            distances = []
            points = list(data.loc[record, :])
            for centroid in centroids:
                cent = list(centroid)
                dist = np.sqrt(sum([(p - c) ** 2 for p, c in zip(points, cent)]))
                distances.append(dist)
            cluster_num[record] = np.argmin(distances)

        return cluster_num

    def _shift_centroids(self, centroids, data):
        new_centroids = []
        n_features = data.shape[1] - 1
        for number, centroid in enumerate(centroids):
            num_k_data = data[data['cluster num'] == number]
            cent_new = []
            for i in range(n_features):
                cent = np.mean(num_k_data[i]) if len(num_k_data) != 0 else np.mean(data[i])
                cent_new.append(cent)
            cent_new = tuple(cent_new)
            new_centroids.append(cent_new)

        return new_centroids

    def _centroid_shift_dist(self, new, old):
        centroid_shifts = []
        for index in range(len(new)):
            c0 = list(new[index])
            c1 = list(old[index])
            shift = np.sqrt(sum([(x - y)**2 for x, y in zip(c0, c1)]))
            centroid_shifts.append(shift)
        less_tol = [elm < self.tolerance for elm in centroid_shifts]

        return less_tol
