import os
import Prepocesser
from sklearn import svm

x_train = Prepocesser.fetch_author_tweets_tokens(False)
y_train = Prepocesser.fetch_author_truths(False)[1]
x_test = Prepocesser.fetch_author_tweets_tokens(True)
y_test = Prepocesser.fetch_author_truths(True)[1]

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


#populate_svm_file(False, x_train, y_train)
#train_svm()
#populate_svm_file(True, x_test, y_test)
#classify_svm()

ovoSVM = svm.SVC(C=10)
ovoSVM.fit(x_train, y_train)

ovoCorrect = 0
total = 0

ovoPredictions = ovoSVM.predict(x_test)

for i in range(len(x_test)):
    if ovoPredictions[i] == y_test[i]:
        ovoCorrect += 1
    total += 1

print("ovo", ovoCorrect/total)

