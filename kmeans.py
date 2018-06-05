import numpy as np
import time
from sklearn.metrics import silhouette_score


class KMeans:

    def __init__(self, dataset: np.array, k: int):
        start = time.time()

        self.k = k
        self.dataset = dataset
        sample_indexes = np.random.choice(dataset.shape[0], k, replace=False)
        self.centers = dataset[sample_indexes]

        self.classes = np.zeros((dataset.shape[0],), dtype=int)

        self.complete = False

        self.start()

        self.silhouette = silhouette_score(dataset, self.classes, metric='euclidean')

        end = time.time()

        self.time = end - start

    def euclidean(self, x: np.array, y: np.array):
        return np.sqrt((np.sum((x - y)**2)))

    def clustering(self):
        for i in range(0, self.dataset.shape[0]):
            closest = self.euclidean(self.dataset[i], self.centers[0])
            cluster_id = 0
            for j in range(1, self.k):
                current = self.euclidean(self.dataset[i], self.centers[j])
                if(current < closest):
                    closest = current
                    cluster_id = j
            self.classes[i] = cluster_id
        self.update_centers()

    def update_centers(self):
        old_centers = np.copy(self.centers)
        for i in range(0, self.k):
            indexes = np.where(self.classes == i)
            self.centers[i] = np.mean(self.dataset[indexes], 0)
        if(self.euclidean(old_centers, self.centers) <= 1e-100):
            self.complete = True

    def start(self):
        while(not(self.complete)):
            self.clustering()
