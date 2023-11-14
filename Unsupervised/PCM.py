import numpy as np
import pandas as pd


# A possibilistic approach to clustering; possibilistic c-means (PCM)
class PCM:
    def __init__(self, data, num_clusters, m=2, max_epochs=1000, tol=1e-2):
        self.data = data
        self.num_clusters = num_clusters
        self.m = m
        self.max_epochs = max_epochs
        self.tol = tol
        self.centers = data[np.random.choice(data.shape[0], num_clusters, replace=False)]
        self.membership = None

    def fit(self):
        for _ in range(self.max_epochs):
            old_centers = self.centers.copy()

            self.membership = self.update_membership()

            self.centers = self.update_center()

            # convergence ?
            if np.linalg.norm(self.centers - old_centers) < self.tol:
                break

        print('Done fitting PCM')
        return self.centers, self.membership

    def update_membership(self):
        membership = np.zeros((self.data.shape[0], self.num_clusters))

        for i in range(self.data.shape[0]):
            for j in range(self.num_clusters):
                numerator = np.linalg.norm(self.data[i] - self.centers[j])
                denominator = self.tol

                for k in range(self.num_clusters):
                    dist = np.linalg.norm(self.data[i] - self.centers[k])
                    dist = np.maximum(dist, 1e-8)  # Replace zeros with 1e-8
                    denominator += (numerator / dist) ** (1 / (self.m - 1))

                membership[i][j] = 1 / denominator

        return membership

    def update_center(self):
        centers = np.zeros((self.num_clusters, self.data.shape[1]))

        for j in range(self.num_clusters):
            denominator = np.sum(self.membership[:, j] ** self.m)

            for k in range(self.data.shape[0]):
                centers[j] += (self.membership[k, j] ** self.m) * self.data[k]

            centers[j] /= denominator

        return centers


if __name__ == '__main__':
    print('hey')
