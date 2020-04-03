import os
import Prepocesser


def train_svm():
    open("svm/model", "w+")
    os.system("svm_multiclass_learn -c 100.0 -e 0.01 svm/train.dat svm/model")
    return True


def classify_svm():
    open("svm/predictions", "w+")
    os.system("svm_multiclass_classify svm/test.dat svm/model svm/predictions")
    return True


def populate_svm_file(test, features, truth_numbers):
    if test:
        file_name = "svm/test.dat"
    else:
        file_name = "svm/train.dat"
    file = open(file_name, "w+")
    for j, vector in enumerate(features):
        text_vector = int(truth_numbers[j].item() + 1).__str__() + " "
        for i, feature in enumerate(vector):
            if feature != 0:
                text_vector += (i+1).__str__() + ":" + int(feature.item()).__str__() + " "
        file.write(text_vector + "\n")


populate_svm_file(False, Prepocesser.fetch_author_tweets_tokens(False), Prepocesser.fetch_author_truths(False)[0])
train_svm()
populate_svm_file(True, Prepocesser.fetch_author_tweets_tokens(True), Prepocesser.fetch_author_truths(True)[0])
classify_svm()

