import pickle
import argparse
from helper_fns import data_read_convert_to_np_array, split_train_test, preprocess_text_data, read_label_from_text_file
from sklearn.metrics import f1_score, accuracy_score


parser = argparse.ArgumentParser()
parser.add_argument('-test_file', '--test-data', help='path of test file', required=False)
parser.add_argument('-test_file_label', '--test-label', help='path of test label file', required=False)
parser.add_argument('-dataset', '--dataset', help='data file name Dolphins,PubMed, Twitter', required=False)

args = vars(parser.parse_args())
# print(args.keys())
"""
test-data has been converted to test_data
test-label has been converted to test_label
"""
if args["dataset"] == "Twitter" or args["dataset"] == "twitter":
    twit_data = preprocess_text_data(args["test_data"])
    twit_label = read_label_from_text_file(args["test_label"])
    twit_label += 1   ###in implementation of BAYEs labels should be 0,1,2,3....
    with open("twitter_clf.pickle" , "rb") as f:
        clf = pickle.load(f)
    pred_lab = clf.predict(twit_data)
    acc = accuracy_score(twit_label, pred_lab)
    macro = f1_score(twit_label, pred_lab, average="macro")
    micro = f1_score(twit_label, pred_lab, average="micro")
    print("Test Accuracy :: " + str(acc * 100))
    print("Test Macro F1-score :: " + str(macro * 100))
    print("Test Micro F1-score :: " + str(micro * 100))
    pass
elif args["dataset"] in ["PubMed", "pubmed","Pubmed"]:
    pubmed_data = data_read_convert_to_np_array(args["test_data"])
    pubmed_label = data_read_convert_to_np_array(args["test_label"])
    with open("pubmed_clf.pickle" , "rb") as f:
        clf = pickle.load(f)
    pred_lab = clf.predict(pubmed_data)
    acc = accuracy_score(pubmed_label, pred_lab)
    macro = f1_score(pubmed_label, pred_lab, average="macro")
    micro = f1_score(pubmed_label, pred_lab, average="micro")
    print("Test Accuracy :: "+ str(acc*100))
    print("Test Macro F1-score :: " + str(macro * 100))
    print("Test Micro F1-score :: " + str(micro * 100))
    pass
elif args["dataset"] in ["dolphins", "Dolphins", "dolphin", "Dolphin"]:
    dolph_data = data_read_convert_to_np_array(args["test_data"])
    dolph_label = data_read_convert_to_np_array(args["test_label"])
    with open("dolph_clf.pickle", "rb") as f:
        clf = pickle.load(f)
    pred_lab = clf.predict(dolph_data)
    acc = accuracy_score(dolph_label, pred_lab)
    macro = f1_score(dolph_label, pred_lab, average="macro")
    micro = f1_score(dolph_label, pred_lab, average="micro")
    print("Test Accuracy :: " + str(acc * 100))
    print("Test Macro F1-score :: " + str(macro * 100))
    print("Test Micro F1-score :: " + str(micro * 100))
    pass
else:
    print("dataset should be \n ")
    print("PubMed pubmed Pubmed")
    print("dolphins Dolphins dolphin Dolphin")
    print("Twitter Twitter")

