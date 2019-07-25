from sklearn.preprocessing import normalize
from sklearn.cluster import DBSCAN
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


class DBScan_nD:
    """Clustering method using the DBSCAN algorithm.

    Parameters
    ----------
    eps : int/float
        Euclidian distance to look around a randomly selected point.
    min_points : int
        Minimum number of points within eps distance to begin clustering off the set of points.

    """

    def __init__(self, eps=0.1, min_points=5):
        self.eps = eps
        self.min_points = min_points

    def fit(self, dbscan_data):
        t0 = time.time()
        """Fit the model to the data passed in.

        Parameters
        ----------
        dbscan_data : Pandas DataFrame
            Pandas DataFrame with n_features, each feature being a single column.

        Returns
        -------
        self
            class like object to call methods and properties off of.

        """
        eps = self.eps
        min_points = self.min_points
        n_features = dbscan_data.shape[1] # num of columns of the dataset
        noise = -1 # noise value (used for coloring points in graphs)
        # cluster_num = [None for i in range(len(dbscan_data))]
        cluster_num = [None] * len(dbscan_data) # no indexes belong to any clusters
        not_visited_indexes = [i for i, val in enumerate(cluster_num) if val is None] # which indexes have not been visited yet
        visited_indexes = [i for i, val in enumerate(cluster_num) if val is not None] # which indexes have been visited

        i = 0 # set initial cluster to 0
        while len(not_visited_indexes) != 0: # while there are still unvisited points
            start_index = np.random.choice(not_visited_indexes, size=1)[0] # choose random staring index
            point_start = [dbscan_data.loc[start_index, i] for i in range(n_features)] # get coordinates of that point

            in_cluster_index = [] # elements in the cluster

            within_eps = dbscan_data[np.sqrt(
                sum([(dbscan_data[i] - point_start[i]) ** 2 for i in range(n_features)])) <= eps].index # get the data that is within eps eclidean distance (close)

            if len(within_eps) >= min_points: # if num points within eps is larger than defined threshold
                visited_indexes.append(start_index) # add the index to ones that have been visited
                not_visited_indexes.remove(start_index) # remove it from the ones that are yet to be visited

                in_cluster_index.extend(within_eps) # add all those within eps to the cluster
                unchecked_in_cluster = list(set(in_cluster_index) - set(visited_indexes)) # which points within the current cluster have not been visited yet

                for index in unchecked_in_cluster: # iterate over the unchecked points in that cluster

                    point = [dbscan_data.loc[index, i] for i in range(n_features)] # list of data points

                    visited_indexes.append(index) # has been vsited
                    not_visited_indexes.remove(index)

                    within_eps = dbscan_data[np.sqrt(
                        sum([(dbscan_data[i] - point[i]) ** 2 for i in range(n_features)])) <= eps].index # get all points within eps

                    unique_indexes = list(
                        set(within_eps) - (set(visited_indexes).union(unchecked_in_cluster))) # remove points that are within eps but are a;ready in the cluster

                    in_cluster_index.extend(within_eps) # add all within eps to the ones in cluster
                    unchecked_in_cluster.extend(unique_indexes) # add the ones that have not been assigned to ones that need checking

                for index in in_cluster_index:
                    cluster_num[index] = i # assign to cluster i

                i += 1 # move onto next cluster
            else: # if there are fewer than min_points within eps
                visited_indexes.append(start_index) # visited
                not_visited_indexes.remove(start_index)
                cluster_num[start_index] = noise # mark as noise

        self.classification = np.array(cluster_num) # array of cluster nums added to class variable
        t1 = time.time()
        self.fit_runtime = t1 - t0
        return self

    @property
    def n_clusters(self):
        return len(set(self.classification) - set([-1])) # num cluster without outlier counting as cluster

    @property
    def outlier_count(self):
        return len([i for i in self.classification if i == -1]) # num outliers in dataset
