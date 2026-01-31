from abc import ABC, abstractmethod
from typing import Any, Dict, Optional
from numpy.typing import NDArray
import numpy as np


class Clusterer(ABC):
    def __init__(self):
        self.model: Any = None
        self.X: NDArray[np.floating] = None
        
    @abstractmethod
    def fit(self, X: NDArray[np.floating]) -> None:
        """
        Fit model to the data

        param: X: (N x D) np.ndarray of data points
        """
        ...

    @abstractmethod
    def predict(self) -> NDArray[np.intp]:
        """
        Generates clusters and returns label for each point
        
        return: labels: (N, ) np.ndarray of labels for each data point
        """
        ...

    def fit_predict(self, X: NDArray[np.floating]) -> NDArray[np.intp]:
        """ 
        Convenience function for combining fit() and predict()

        param: X: (N x D) np.ndarray of data points
        return: labels: (N, ) np.ndarray of labels for each data point
        """
        self.fit(X)
        return self.predict()


    def get_cluster_stats(self, cluster: NDArray[np.floating]) -> Dict[str, Any]:
        """
        Generates dictionary of some metadata of a given cluster

        param: cluster: (N x D) np.ndarray of data points
        return: Dict[str, Any] containing metadata for a single cluster
        """
        n = int(cluster.shape[0])
        centroid = cluster.mean(axis=0)
        mean_std = float(cluster.std(axis=0).mean())
        diffs = cluster - centroid   #used to calculate medoid index only
        sse = float((diffs * diffs).sum())  #used to calculate medoid index only
        dists = np.sqrt(((cluster[:, None, :] - cluster[None, :, :]) ** 2).sum(axis=2))
        medoid_idx = int(np.argmin(dists.sum(axis=1)))
        return {
            "n_points": n,
            "centroid": centroid,
            "mean_std": mean_std,
            "sse": sse,
            "medoid_idx": medoid_idx,
        }