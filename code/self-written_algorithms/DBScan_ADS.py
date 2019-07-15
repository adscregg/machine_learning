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
        n_features = dbscan_data.shape[1]
        noise = -1
        # cluster_num = [None for i in range(len(dbscan_data))]
        cluster_num = [None] * len(dbscan_data)
        not_visited_indexes = [i for i, val in enumerate(cluster_num) if val is None]
        visited_indexes = [i for i, val in enumerate(cluster_num) if val is not None]

        i = 0
        while len(not_visited_indexes) != 0:
            start_index = np.random.choice(not_visited_indexes, size=1)[0]
            point_start = [dbscan_data.loc[start_index, i] for i in range(n_features)]

            in_cluster_index = []

            within_eps = dbscan_data[np.sqrt(
                sum([(dbscan_data[i] - point_start[i]) ** 2 for i in range(n_features)])) <= eps].index

            if len(within_eps) >= min_points:
                visited_indexes.append(start_index)
                not_visited_indexes.remove(start_index)

                in_cluster_index.extend(within_eps)
                unchecked_in_cluster = list(set(in_cluster_index) - set(visited_indexes))

                for index in unchecked_in_cluster:

                    point = [dbscan_data.loc[index, i] for i in range(n_features)]

                    visited_indexes.append(index)
                    not_visited_indexes.remove(index)

                    within_eps = dbscan_data[np.sqrt(
                        sum([(dbscan_data[i] - point[i]) ** 2 for i in range(n_features)])) <= eps].index

                    unique_indexes = list(
                        set(within_eps) - (set(visited_indexes).union(unchecked_in_cluster)))

                    in_cluster_index.extend(within_eps)
                    unchecked_in_cluster.extend(unique_indexes)

                for index in in_cluster_index:
                    cluster_num[index] = i

                i += 1
            else:
                visited_indexes.append(start_index)
                not_visited_indexes.remove(start_index)
                cluster_num[start_index] = noise

        self.classification = np.array(cluster_num)
        t1 = time.time()
        self.fit_runtime = t1 - t0
        return self

    @property
    def n_clusters(self):
        return len(set(self.classification) - set([-1]))

    @property
    def outlier_count(self):
        return len([i for i in self.classification if i == -1])
