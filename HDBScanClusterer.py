from __future__ import annotations
from typing import Any, Dict
import numpy as np
from numpy.typing import NDArray
from sklearn.cluster import HDBSCAN
from Clusterer import Clusterer


class HDBScanClusterer(Clusterer):
    def __init__(self, **hdb_kwargs: Any) -> None:
        super().__init__()
        self.model = HDBSCAN(
            store_centers="both", 
            **hdb_kwargs,
        )

    def fit(self, X: NDArray[np.floating]) -> None:
        self.model.fit(X)
        return
    
    def predict(self) -> NDArray[np.intp]:
        return self.model.labels_
