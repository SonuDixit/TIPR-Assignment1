"""
Created on Sun Jan 27 10:42:01 2019
@author: sonu
"""
import numpy as np
def get_ML_params(data, is_categorical = False):
    """
    :param data: is a 1d np array, 1 column of data matrix
    :param is_categorical:
    :return:
    """
    if not is_categorical :
        mean = np.mean(data)
        std = np.std(data)
        return mean,std
    else:
        all_uniques = list(set(list(data)))
        all_uniques.sort()
        multinomial_params = {}
        for i in range(len(all_uniques)):
            multinomial_params[all_uniques[i]] = list(data).count(all_uniques[i]) / len(list(data))
        return multinomial_params
def get_prob(data,params,is_cat= False):
    if not is_cat:
        # params[0] = mean, params[1] = std
        log_density = np.log(np.exp(-0.5 * (((data-params[0])/params[1]) ** 2.0)) * 1/(params[1] * np.sqrt(2*np.pi)))
        return log_density
    else:
        # print("number of params are", len(params))
        # print("implement for multinomial distribution")
        #params is now a dictionary
        if data not in params.keys():
            print("keys are ",params.keys())
            print("data not in key of dict, data=",data)
        return np.log(params[data])

class BayesClf:
    def __init__(self):
        pass
    def fit(self,data_mat,label,bow=False, is_categorical=None):
        """
        data_mat is n x feature
        label is 1d array, assuming it contains 0,1,2....numlabels-1.
        is_categorical is a list of length data_mat.shape[1]
        estimate prior
        estimate class conditional density:
            continous -> mean and variance
            discreete -> just count
        """
        label = list(label)

        self.prior=[]
        self.params = []
        self.bow = bow
        for i in range(len(list(set(label)))):
            self.prior.append(label.count(i)/len(label))

        unique_label = list(set(label))
        unique_label.sort()

        if self.bow:
            laplace_smooth = data_mat.shape[1]
            for i in unique_label:
                """
                find rows corresponding to this label
                """
                row_index = [j for j in range(len(label)) if i == label[j]]
                class_conditional_data = data_mat[row_index, :]
                num_sum = np.sum(class_conditional_data, axis=0)
                deno = np.sum(num_sum)
                cc_params = (num_sum + 1) / (deno+laplace_smooth)
                self.params.append(cc_params)

            return self.prior, self.params

        if is_categorical is None:
            is_categorical = [False] * data_mat.shape[1]
        self.is_categorical = is_categorical

        for i in unique_label:
            """
            find rows corresponding to this label
            """
            row_index = [j for j in range(len(label)) if i == label[j]]
            class_conditional_data = data_mat[row_index,:]
            cc_params =[]
            for j in range(data_mat.shape[1]):
                cc_params.append(get_ML_params(class_conditional_data[:,j],is_categorical[j]))
            self.params.append(cc_params)

        return self.prior, self.params

    def predict(self,test_data_mat):
        """
        :param test_data_mat: is a numpy 2d array num_examples X features
        :return:
        """
        predict_label = []
        if self.bow:
            for i in range(test_data_mat.shape[0]):
                log_prob = [np.log(self.prior[j]) for j in range(len(self.prior))]
                for k in range(len(self.prior)):  # all classes
                    s = self.params[k] ** test_data_mat[i]
                    s = np.log(s)
                    s = np.sum(s)
                    log_prob[k] += s
                predict_label.append(np.argmax(log_prob))
            return predict_label

        for i in range(test_data_mat.shape[0]):
            log_prob = [np.log(self.prior[j]) for j in range(len(self.prior))]
            for k in range(len(self.prior)):#all classes
                s = 0
                for m in range(len(self.params)):#all_features
                    s += get_prob(test_data_mat[i][m],self.params[k][m], self.is_categorical[m])
                log_prob[k] += s
            predict_label.append(np.argmax(log_prob))

        return predict_label