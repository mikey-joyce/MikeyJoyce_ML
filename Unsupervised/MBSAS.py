import numpy as np
import math
from statistics import mode
from BSAS import BSAS

class MBSAS(BSAS):
    def __init__(self):
        BSAS.__init__(self)

    def fit(self, data, thresh=1, max_clusters=math.inf):
        m = 0  # initialize cluster counter
        clusters = [[data[0]]]  # initialize cluster array
        self.centers = [data[0]]
        membership = [0]

        # Cluster determination
        for i in range(1, data.shape[0]):
            # pick the closest cluster
            distance, _ = self.__find_cluster(data, clusters, i)

            # determine if a new cluster needs to be created
            if distance > thresh and m < (max_clusters - 1):
                m += 1
                clusters.append([data[i]])
                self.centers.append(data[i])
                membership.append(m)
            else:
                # create flag that lets us know this data point has not been assigned to a cluster
                membership.append(False)

        # Pattern classification
        for i in range(data.shape[0]):
            if membership[i] == False:
                # pick the closest cluster
                distance, k = self.__find_cluster(data, clusters, i)

                clusters[k].append(data[i])
                # recompute center
                size = len(clusters[k])
                self.centers[k] = ((size - 1) * self.centers[k] + data[i]) / size
                membership.append(k)
