import numpy as np
from helper_fns import bin_to_decimal
class LSH:
    def __init__(self,data_mat):
        """
        :param data_mat: is a numpy 2d array, num_examples X feature dimension
        """
        self.data_mat = data_mat


    def fit(self,num_bands,functions_per_band=8):
        self.num_bands = num_bands
        self.functions_per_band = functions_per_band
        np.random.seed(1234)
        self.random_vectors = np.random.normal(0,1,(num_bands*functions_per_band , self.data_mat.shape[1]))
        self.sim_table = np.dot(self.data_mat,self.random_vectors.T)
        self.sim_table[np.where(self.sim_table > 0)] = 1
        self.sim_table[np.where(self.sim_table <= 0)] = 0
        # print(self.sim_table)

        self.hash_table = {}
        self.red_dim = np.zeros((self.data_mat.shape[0],num_bands))
        ## key will be decimal number, value will be index of data in data_mat
        for i in range(num_bands):
            cols = range(i*functions_per_band, (i+1)*functions_per_band )
            data = self.sim_table[:,cols]
            for j in range(data.shape[0]):
                k = bin_to_decimal(list(data[j]))
                self.red_dim[j][i] = k
                if k not in self.hash_table:
                    self.hash_table[k] = {}
                    self.hash_table[k][j] = 1
                else:
                    if j not in self.hash_table[k]:
                        self.hash_table[k][j] = 1
                    else:
                        self.hash_table[k][j] += 1
        return self.red_dim

    def predict_nn(self, data_points, true_label_as_list):
        """
        find the nearest neighbor of data_point
        :param data_point: 2d numpy array
        :return:
        """
        pred_dicts = [{} for _ in range(data_points.shape[0])]
        sim_table = np.dot(data_points, self.random_vectors.T)
        sim_table[np.where(sim_table > 0)] = 1
        sim_table[np.where(sim_table <= 0)] = 0
        for i in range(self.num_bands):
            cols = range(i*self.functions_per_band, (i+1)*self.functions_per_band )
            data = sim_table[:,cols]
            for j in range(data.shape[0]):
                k = bin_to_decimal(list(data[j]))
                if k not in self.hash_table:
                    print("this key was not seen during fit: ",str(k))

                else:
                    #find max_count key pair
                    max = 0
                    for key,count in self.hash_table[k].items():
                        if count>max:
                            max = count

                    for key,count in self.hash_table[k].items():
                        if count == max:
                            if key in pred_dicts[j]:
                                if pred_dicts[j][key] < max:
                                    pred_dicts[j][key] = max
                            else:
                                pred_dicts[j][key] = max

        result_arr = np.zeros(data_points.shape)
        labels = []
        for i in range(data_points.shape[0]):
            max = 0
            k = -2
            for key,val in pred_dicts[i].items():
                if val>max:
                    max = val
                    result_arr[i] = self.data_mat[key]
                    k = key
            labels.append(true_label_as_list[k])


        return  result_arr,labels



# data_mat = np.asarray([[1,2],[3,4]])
# lsh = LSH(data_mat)
# lsh.fit(2,3)