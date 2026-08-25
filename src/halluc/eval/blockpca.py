"""Per-block PCA, so each method contributes equally to the union model.

A plain concatenation gives LapEigvals 16,000 of ~21,200 columns; any advantage the
union shows could then be dimensionality rather than complementarity. Reducing every
block to the same width first removes that explanation.
"""

from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.decomposition import PCA


class BlockPCA(BaseEstimator, TransformerMixin):
    def __init__(self, slices=None, n_components: int = 128, random_state: int = 0) -> None:
        self.slices = slices
        self.n_components = n_components
        self.random_state = random_state

    def fit(self, X, y=None):
        X = np.asarray(X)
        self.pcas_ = []
        for block in self.slices:
            width = block.stop - block.start
            # A block narrower than the target (ICR is 40 dims, SVD 41) is passed
            # through untouched rather than padded.
            components = min(self.n_components, width, X.shape[0])
            if components >= width:
                self.pcas_.append(None)
            else:
                pca = PCA(n_components=components, random_state=self.random_state)
                pca.fit(X[:, block])
                self.pcas_.append(pca)
        return self

    def transform(self, X):
        X = np.asarray(X)
        parts = [
            X[:, block] if pca is None else pca.transform(X[:, block])
            for block, pca in zip(self.slices, self.pcas_)
        ]
        return np.concatenate(parts, axis=1)
