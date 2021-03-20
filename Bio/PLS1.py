import warnings
from abc import ABCMeta, abstractmethod

import numpy as np
from scipy.linalg import pinv2, svd

class PLSRegression:

    def _get_first_singular_vectors_power_method(self, X, Y, max_iter=500,
                                                 tol=1e-06):

        eps = np.finfo(X.dtype).eps
        y_score = next(col for col in Y.T if np.any(np.abs(col) > eps))
        x_weights_old = 100  # init to big value for first convergence check


        for i in range(max_iter):

                x_weights = np.dot(X.T, y_score) / np.dot(y_score, y_score)

            x_weights /= np.sqrt(np.dot(x_weights, x_weights)) + eps
            x_score = np.dot(X, x_weights)
            x_weights_diff = x_weights - x_weights_old
            if np.dot(x_weights_diff, x_weights_diff) < tol or Y.shape[1] == 1:
                break
            x_weights_old = x_weights

        n_iter = i + 1
    return x_weights, n

    def _center_scale_xy(self ,X, Y, scale=True):
        # center
        x_mean = X.mean(axis=0)
        X -= x_mean
        y_mean = Y.mean(axis=0)
        Y -= y_mean
        # scale
        if scale:
            x_std = X.std(axis=0, ddof=1)
            x_std[x_std == 0.0] = 1.0
            X /= x_std
            y_std = Y.std(axis=0, ddof=1)
            y_std[y_std == 0.0] = 1.0
            Y /= y_std
        else:
            x_std = np.ones(X.shape[1])
            y_std = np.ones(Y.shape[1])
        return X, Y, x_mean, y_mean, x_std, y_std

    def fit(self, X, Y, components):
        n = X.shape[0]
        p = X.shape[1]
        q = 1
        nComponents = components

        self._center_scale_xy(X,Y)

        self.x_weights_ = np.zeros((p, nComponents))  # U
        #self.y_weights_ = np.zeros((q, nComponents))  # V
        self._x_scores = np.zeros((n, nComponents))  # Xi
        self.x_loadings_ = np.zeros((p, nComponents))  # Gamma
        #self.y_loadings_ = np.zeros((q, nComponents))  # Delta
        self.n_iter_ = []

        Y_eps = np.finfo(Y.dtype).eps
        for k in range(nComponents):
            #alghorithm nipals
