"""
this contains some helper fns like data read file
"""
import os
import pandas as pd
import numpy as np
import pickle
import random

def data_read_convert_to_np_array(data_path, sep =" ", header=None):
    df = pd.read_csv(data_path,sep=sep,header=header)
    np_arr = df.values
    if np_arr.shape[1] == 1:
        np_arr = np_arr.reshape((np_arr.shape[0],))
    return np_arr

def split_train_test(np_2d_data, np_1d_label, seed = 1234, test=0.20):
    random.seed(seed)
    test_index = random.sample(range(np_2d_data.shape[0]), int(np_2d_data.shape[0] * test))
    train_index = [i for i in range(np_2d_data.shape[0]) if i not in test_index]
    test_data = np_2d_data[test_index,:]
    test_label = np_1d_label[test_index]
    train_data = np_2d_data[train_index,:]
    train_label = np_1d_label[train_index]
    return train_data, train_label, test_data, test_label

def preprocess_text_data(text_file_path, word_to_index_pickle_path= os.path.join(os.path.dirname(os.getcwd()),"data","twitter","word_to_index.pickle")):
    with open(text_file_path,"r") as f:
        lines = f.readlines()
    good_lines = [line.rstrip("\n") for line in lines]

    with open(word_to_index_pickle_path,"rb") as f:
        word_index_dict = pickle.load(f)

    data = np.zeros((len(good_lines),len(word_index_dict)))

    for i in range(len(good_lines)):
        # print(good_lines[i])
        t = good_lines[i].split(" ")
        for j in range(len(t)):
            if t[j] in word_index_dict:
                data[i][word_index_dict[t[j]]] += 1
    return data

def read_label_from_text_file(label_file):
    with open(label_file,"r") as f:
        lines = f.readlines()
    labels = [int(line.rstrip("\n")) for line in lines ]
    return np.asarray(labels)

def bin_to_decimal(bin_lst):
    d = bin_lst[0]
    for i in bin_lst[1:]:
        d = d*2 + i
    return d