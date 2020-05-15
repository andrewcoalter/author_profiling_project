import Preprocesser
from sklearn import svm

x_train = Preprocesser.fetch_author_tweets_tokens(False, False)
y_train_0, y_train_1 = Preprocesser.fetch_author_truths(False, False)
x_test = Preprocesser.fetch_author_tweets_tokens(True, True)
y_test_0, y_test_1 = Preprocesser.fetch_author_truths(True, True)



def SVM_run(C, gamma, kernel, age):

    if age:
        y = y_train_0
    else:
        y = y_train_1
    ovoSVM = svm.SVC(C=C, gamma=gamma, kernel=kernel, decision_function_shape="ovr")
    ovoSVM.fit(x_train, y)

    ovoCorrect = 0
    total = 0

    ovoPredictions = ovoSVM.predict(x_test)

    if age:
        y_test = y_test_0
    else:
        y_test = y_test_1
    for i in range(len(x_test)):
        if ovoPredictions[i] == y_test[i]:
            ovoCorrect += 1
        total += 1

    return round((ovoCorrect/total), 3), ovoSVM

