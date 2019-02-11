import numpy as np

class NearestNeighbour:
    def __init__(self, k=1):

        self.k = k

    def fit(self, data_matrix, label):
        """
        data_matrix is num_examples X feature_dimension numpy array
        label is 1d array
        """
        self.train_data = data_matrix
        self.train_label = label
        pass
    def predict(self, test_data):
        """
        test-data is num-examples X feature_dimension numpy array
        current implementation is 1-nearest neighbor
        if test_data is only one example, convert it to 2d array
        """
        test_label = np.zeros((test_data.shape[0], ))
        if self.k == 1:
            for i in range(test_data.shape[0]):
                temp = self.train_data - test_data[i, :]
                temp = np.square(temp)
                dist = np.sum(temp, axis=1)
                # print("dist is", dist)
                # print(np.argmin(dist))
                # print(self.train_label[np.argmin(dist)])
                test_label[i] = self.train_label[np.argmin(dist)]
            return test_label
        else:
            for i in range(test_data.shape[0]):
                temp = self.train_data - test_data[i, :]
                temp = np.square(temp)
                dist = np.sum(temp, axis=1)
                indices = np.argpartition(dist,self.k)
                dict_count = {}
                for j in range(self.k):
                    if self.train_label[indices[j]] not in dict_count.keys():
                        dict_count[self.train_label[indices[j]]] = 1
                    else:
                        dict_count[self.train_label[indices[j]]] += 1
                v_max = 0
                for key,v in dict_count.items():
                    if v>v_max:
                        v_max = v
                        test_label[i] = key
            return test_label
            # print("implement for k>1 KNN.")

    def f1_calculate_micro(self, predict_label, true_label):
        """
        :param actual_label: 1d numpy array, morethan 2 class
        :param true_label: 1d numpy array,
        :return: f1 score, precision, recall, all are sme for micro calculation
        """
        corr = 0
        for i in range(true_label.shape[0]):
            if predict_label[i] == true_label[i]:
                corr += 1
        return corr / true_label.shape[0]

    def f1_calculate_one_vs_all(self, predict_label, true_label):
        """
        :param actual_label: 1d numpy array, 1/0, 1:positive, 0:negative
        :param true_label: 1d numpy array, 1/0, 1:positive, 0: negative
        :return: f1 score

        precision = TP/(TP+FP)
        recall = TP/(TP+FN)
        F1 = 2*precision*recall /(precision+recall)
        """
        TP=0
        FP=0
        FN = 0
        for i in range(true_label.shape[0]):
            if predict_label[i]==1:
                if true_label[i] == 1:
                    TP+=1
                else:
                    FP+=1
            elif true_label[i] == 1:
                FN +=1
        precision = TP / (TP + FP)
        recall = TP / (TP + FN)
        F1 = 2 * precision * recall / (precision + recall)
        return F1

    def f1_calculate_macro(self, test_data,test_actual_label):
        """
        write code for f1 calculation for both macro and micro
        """
        test_label = self.predict(test_data)
        #for each class call f1_one_vs all
        # take their average

        pass
