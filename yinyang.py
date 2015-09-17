import numpy as np

from sklearn.base import BaseEstimator, ClusterMixin
from sklearn.utils.extmath import row_norms

class YinyangKMeans(BaseEstimator, ClusterMixin):
    """
    Scikit-learn compatible K-Means clusterer based on
    http://jmlr.org/proceedings/papers/v37/ding15.pdf
    """

    def __init__(self, n_clusters=3, init="random", max_iter=300, tol=0.0001,
                 random_state=None):
        self.n_clusters = n_clusters
        self.init = init
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state

    def fit(self, X):
        rng = np.random.RandomState(self.random_state)
        new_cluster_centers = np.zeros((self.n_clusters, X.shape[1]))
        n_samples_arrays = np.arange(X.shape[0])

        if self.n_clusters > 20:
            raise ValueError("Group clustering not supported yet")

        if self.init == "random":
            old_cluster_centers_ = X[rng.randint(0, X.shape[0], self.n_clusters), :]
        else:
            raise ValueError("wait till we support other initializations.")

        # Run K-Means for the first time.
        # Don't do cluster.KMeans().fit(X) because of input_validation etc.
        dot_product = 2 * np.dot(X, old_cluster_centers_.T)
        cluster_norms = row_norms(old_cluster_centers_, squared=True).reshape(1, -1)
        self.distances_ = row_norms(X, squared=True).reshape(-1, 1) - dot_product + cluster_norms

        # Remove the closest and the second closest cluster.
        upper_and_lower_bounds = np.argpartition(self.distances_, 1, axis=1)
        self.labels_ = upper_and_lower_bounds[:, 0]
        self.almost_labels_ = upper_and_lower_bounds[:, 1]
        self.upper_and_lower_bounds_ = self.distances_[n_samples_arrays.reshape(-1, 1), upper_and_lower_bounds]

        # Update cluster centers
        for i in range(self.n_clusters):
            new_cluster_centers[i] = np.mean(X[self.labels_ == i], axis=0)
        self.cluster_centers_ = new_cluster_centers

        for n_iter in range(self.max_iter):

            # Calculate how much each center has drifted.
            drift = ((old_cluster_centers_ - self.cluster_centers_)**2).sum(axis=1)
            if np.sum(drift) < self.tol:
                break
            old_cluster_centers_ = np.copy(self.cluster_centers_)

            # Add the drift to the upper bounds and subtract the drift from the lower bounds.
            for i in range(self.n_clusters):
                mask = self.labels_ == i
                self.upper_and_lower_bounds_[:, 0][mask] += drift[i]
                self.upper_and_lower_bounds_[:, 1][mask] -= drift[i]

            # If the previously second_largest_bound is now lesser than the largest bound
            # set the upper bound to the distance between the largest_bound
            # This is based on d(old_center, new_center) + d(old_center, X) > d(X, new_center)
            mask_changed_bounds = self.upper_and_lower_bounds_[:, 1] < self.upper_and_lower_bounds_[:, 0]

            #XXX: Vectorize?
            for i in range(self.n_clusters):
                cluster = self.cluster_centers_[i]
                new_mask = np.logical_and(mask_changed_bounds, self.labels_ == i)
                distances = np.sum((X[new_mask] - cluster)**2, axis=1)
                self.upper_and_lower_bounds_[:, 0][new_mask] = distances

            # Now we can be sure that the second closest center is actually the closest.
            # Reassign the labels.
            mask_changed_bounds = self.upper_and_lower_bounds_[:, 1] < self.upper_and_lower_bounds_[:, 0]
            tmp = self.labels_[mask_changed_bounds]
            self.labels_[mask_changed_bounds] = self.almost_labels_[mask_changed_bounds]
            self.almost_labels_[mask_changed_bounds] = tmp

            self.upper_and_lower_bounds_[:, 1][mask_changed_bounds] = self.upper_and_lower_bounds_[:, 0][mask_changed_bounds]

            #XXX: Vectorize?
            for i in range(self.n_clusters):
                cluster = self.cluster_centers_[i]
                new_mask = np.logical_and(mask_changed_bounds, self.labels_ == i)
                distances = np.sum((X[new_mask] - cluster)**2, axis=1)
                self.upper_and_lower_bounds_[:, 0][new_mask] = distances

            # TODO: Optimize this step.
            for i in range(self.n_clusters):
                mask = self.labels_ == i
                self.cluster_centers_[i] = np.mean(X[mask], axis=0)

        self.n_iter_ = n_iter
