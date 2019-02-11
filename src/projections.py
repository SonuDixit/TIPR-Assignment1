"""
this is to generate random projection matrix.
"""
import numpy as np
class projections:
    def __init__(self, n_initial_features, n_final_features, seed = 1234):
        self.n_initial_features = n_initial_features
        self.n_final_features = n_final_features
        np.random.seed(seed)
        self.mat = np.random.normal(loc=0,scale=1/n_final_features,
                                    size=(n_final_features,n_initial_features))
    def fit(self,data_mat):
        """
        :param data_mat: np array example X features
        :return:
        """
        return np.matmul(data_mat, self.mat.T)