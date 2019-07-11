if __name__ == '__main__':

    from sklearn.datasets import make_blobs, make_moons, make_circles
    import pandas as pd

    def blobs(num_samples = 1000, n_feats = 3, cents = 3, clust_std = 1.0):
        data = make_blobs(n_samples= num_samples, n_features = n_feats, centers = cents, cluster_std=clust_std)
        X = data[0]
        y = data[1]
        return pd.DataFrame(X), pd.DataFrame(y)

    def moons(num_samples = 1000, noise = 0.05):
        data = make_moons(n_samples= num_samples, noise = noise)
        X = data[0]
        y = data[1]
        return pd.DataFrame(X), pd.DataFrame(y)

    def circles(num_samples = 1000, noise = 0.05, fact = 0.8):
        data = make_circles(n_samples= num_samples, noise = noise, factor = fact)
        X = data[0]
        y = data[1]
        return pd.DataFrame(X), pd.DataFrame(y)
