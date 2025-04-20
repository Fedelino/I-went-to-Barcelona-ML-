import numpy as np
import itertools


class KMeans(object):
    """
    kNN classifier object.
    """

    def __init__(self, max_iters=500, n_init=10, criterion="accuracy", n_clusters=None):
        self.max_iters = max_iters
        self.n_init = n_init
        self.criterion = criterion
        self.n_clusters = n_clusters
        self.centroids = None
        self.best_permutation = None

    def fit(self, training_data, training_labels):
        """
        Trains the model, returns predicted labels for training data.
        Hint:
            (1) Since Kmeans is unsupervised clustering, we don't need the labels for training. But you may want to use it to determine the number of clusters.
            (2) Kmeans is sensitive to initialization. You can try multiple random initializations when using this classifier.

        Arguments:
            training_data (np.array): training data of shape (N,D)
            training_labels (np.array): labels of shape (N,).
        Returns:
            pred_labels (np.array): labels of shape (N,)
        """

        N, D = training_data.shape
        K = self.n_clusters if self.n_clusters is not None else len(np.unique(training_labels))

        best_centroids = None
        best_labels = None
        best_score = -1  
        best_assignments = None
        
        for i in range(self.n_init):  #initialise more times
            # equivalent to d=np.random.permutation(data)
            # centers=d[:K]
            centroids = training_data[np.random.choice(N, K, replace=False)]

            for j in range(self.max_iters):
                distances = np.linalg.norm(training_data[:, np.newaxis, :] - centroids[np.newaxis, :, :], axis=2)
                assignments = np.argmin(distances, axis=1)

                new_centroids = np.array([ #update center
                    training_data[assignments == k].mean(axis=0) if np.any(assignments == k) else centroids[k]
                    for k in range(K) # we could have followed classic for...strcuture, but i prefer this
                ])

                if np.allclose(centroids, new_centroids):
                    break  # convergence
                centroids = new_centroids.copy()

            # We use voting
            cluster_labels = np.zeros(K, dtype=int)
            for k in range(K):
                assigned = training_labels[assignments == k]
                if len(assigned) > 0:
                    cluster_labels[k] = np.bincount(assigned.astype(int)).argmax()

            pred_labels = cluster_labels[assignments]
            
            if self.criterion == "accuracy":
                score = np.mean(pred_labels == training_labels)
            elif self.criterion == "f1":
                from src.utils import macrof1_fn
                score = macrof1_fn(pred_labels, training_labels)
            elif self.criterion == "ssd":
                score = -np.sum((training_data - centroids[assignments]) ** 2)
            else:
                raise ValueError("Invalid scoring: choose from 'accuracy', 'f1', 'ssd'")
            
            # take the best
            if score > best_score:
                best_score = score
                best_centroids = centroids
                best_labels = cluster_labels
                best_assignments = assignments

        self.centroids = best_centroids
        self.best_permutation = best_labels  # maps cluster index → label

        return self.best_permutation[best_assignments]

    def predict(self, test_data):
        """
        Runs prediction on the test data.

        Arguments:
            test_data (np.array): test data of shape (N,D)
        Returns:
            test_labels (np.array): labels of shape (N,)
        """
        distances = np.linalg.norm(test_data[:, np.newaxis, :] - self.centroids[np.newaxis, :, :], axis=2)
        assignments = np.argmin(distances, axis=1)
        test_labels = self.best_permutation[assignments]

        return test_labels
