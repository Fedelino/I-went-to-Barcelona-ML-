import numpy as np

class KNN(object):
    """
    kNN classifier object.
    """

    def __init__(self, k=1, task_kind="classification"):
        self.k = k
        self.task_kind = task_kind
        self.training_data = None
        self.training_labels = None

    def fit(self, training_data, training_labels):
        """
        Since KNN is non-parametric, we just store the data.
        """
        self.training_data = training_data
        self.training_labels = training_labels

        # On retourne les prédictions sur les données d'entraînement
        return self.predict(training_data)

    def predict(self, test_data):
        """
        Predict labels for test_data using k-nearest neighbors.

        Arguments:
            test_data (np.array): (N_test, D)
        Returns:
            test_labels (np.array): (N_test,)
        """
        num_test = test_data.shape[0]
        pred_labels = np.zeros(num_test, dtype=int)

        for i in range(num_test):
            # Compute L2 distances between test[i] and all training samples
            dists = np.linalg.norm(self.training_data - test_data[i], axis=1)

            # Get indices of k nearest neighbors
            knn_idx = np.argsort(dists)[:self.k]
            knn_labels = self.training_labels[knn_idx]

            if self.task_kind == "classification":
                # Vote: most common label
                values, counts = np.unique(knn_labels, return_counts=True)
                pred_labels[i] = values[np.argmax(counts)]

            else:
                # For regression (if needed), average the labels
                pred_labels[i] = np.mean(knn_labels)

        return pred_labels
