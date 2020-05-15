import numpy as np
import torch
import Preprocesser
import Postprocesser
import RNNWrapper
import FFNNWrapper
import LSTMWrapper
import SVMWrapper
import sklearn.metrics as metrics
from sklearn.preprocessing import label_binarize
from scipy.stats import wilcoxon
import matplotlib.pyplot as plt

x_2_social_test, y_2_social_age, y_2_social_gender, social_tweet_numbers = Preprocesser.fetch_tweets_tokens_ordered_with_truths_ind(True, False)
x_2_review_test, y_2_review_age, y_2_review_gender, review_tweet_numbers = Preprocesser.fetch_tweets_tokens_ordered_with_truths_ind(False, True)
x_2_blog_test, y_2_blog_age, y_2_blog_gender, blog_tweet_numbers = Preprocesser.fetch_tweets_tokens_ordered_with_truths_ind(False, False)

x_train, y_train, y_age_train, train_tweet_numbers = Preprocesser.fetch_tweets_tokens_ordered_with_truths(False, False)
x_val, y_val, y_age_val, val_tweet_numbers = Preprocesser.fetch_tweets_tokens_ordered_with_truths(True, True)
x_test, y_test, y_age_test, test_tweet_numbers = Preprocesser.fetch_tweets_tokens_ordered_with_truths(True, False)
x_test_2 = Preprocesser.fetch_author_tweets_tokens(True, False)

x_social_test = Preprocesser.fetch_author_tweets_tokens_ind(True, False)
x_review_test = Preprocesser.fetch_author_tweets_tokens_ind(False, True)
x_blog_test = Preprocesser.fetch_author_tweets_tokens_ind(False, False)
social_y, social_y_2 = Preprocesser.fetch_author_truths_ind(True, False)
review_y, review_y_2 = Preprocesser.fetch_author_truths_ind(False, True)
blog_y, blog_y_2 = Preprocesser.fetch_author_truths_ind(False, False)


def accuracy(y_pred, y):
    correct = 0
    total = 0
    for i in range(len(y_pred)):
        if y_pred[i] == y[i]:
            correct += 1
        total += 1
    return correct/total


def run_postprocessor(model, name):
    h = model.init_hidden(x_train.size(0))
    train_results, _ = model(x_train, h)
    h = model.init_hidden(x_val.size(0))
    val_results, _ = model(x_val, h)
    val_common_acc, val_sum_acc = Postprocesser.author_accuracy(val_results, val_tweet_numbers, 0)
    train = np.zeros([len(train_tweet_numbers), 400])
    train_sum = np.zeros([len(train_tweet_numbers), 4])
    train_true = []
    j = 0
    for a, author_info in enumerate(train_tweet_numbers):
        tweet_number = author_info[0]
        train_true.append(int(author_info[1]))
        counter = 0
        for i in range(int(tweet_number)):
            for k, item in enumerate(train_results.__getitem__(j)):
                train[a][counter] = float(item)
                train_sum[a][k] += float(item)
                counter += 1
            j += 1
    val = np.zeros([len(val_tweet_numbers), 400])
    val_sum = np.zeros([len(val_tweet_numbers), 4])
    val_true = []
    j = 0
    for a, author_info in enumerate(val_tweet_numbers):
        tweet_number = author_info[0]
        val_true.append(int(author_info[1]))
        counter = 0
        for i in range(int(tweet_number)):
            for k, item in enumerate(val_results.__getitem__(j)):
                val[a][counter] = float(item)
                val_sum[a][k] += float(item)
                counter += 1
            j += 1
    train = torch.Tensor(train)
    train_sum = torch.Tensor(train_sum)
    val = torch.Tensor(val)
    val_sum = torch.Tensor(val_sum)
    train_true = torch.tensor(train_true)
    val_true = torch.tensor(val_true)
    best_loss, best_acc, best_epoch, best_hidden_layer_size, best_learning_rate, best_number_of_layers, best_decay_rate = Postprocesser.neural_network(400, train, train_true, val, val_true, name)
    best_2_loss, best_2_acc, best_2_epoch, best_2_hidden_layer_size, best_2_learning_rate, best_2_number_of_layers, best_2_decay_rate = Postprocesser.neural_network(
        4, train_sum, train_true, val_sum, val_true, name + "_2")
    return val_common_acc, val_sum_acc, best_acc, best_2_acc


def run_postprocessor_age(model, name):
    h = model.init_hidden(x_train.size(0))
    train_results, _ = model(x_train, h)
    h = model.init_hidden(x_val.size(0))
    val_results, _ = model(x_val, h)
    val_common_acc, val_sum_acc = Postprocesser.author_accuracy(val_results, val_tweet_numbers, 1)
    train = np.zeros([len(train_tweet_numbers), 200])
    train_sum = np.zeros([len(train_tweet_numbers), 2])
    train_true = []
    j = 0
    for a, author_info in enumerate(train_tweet_numbers):
        tweet_number = author_info[0]
        train_true.append(int(author_info[2]))
        counter = 0
        for i in range(int(tweet_number)):
            for k, item in enumerate(train_results.__getitem__(j)):
                train[a][counter] = float(item)
                train_sum[a][k] += float(item)
                counter += 1
            j += 1
    val = np.zeros([len(val_tweet_numbers), 200])
    val_sum = np.zeros([len(val_tweet_numbers), 2])
    val_true = []
    j = 0
    for a, author_info in enumerate(val_tweet_numbers):
        tweet_number = author_info[0]
        val_true.append(int(author_info[2]))
        counter = 0
        for i in range(int(tweet_number)):
            for k, item in enumerate(val_results.__getitem__(j)):
                val[a][counter] = float(item)
                val_sum[a][k] += float(item)
                counter += 1
            j += 1
    train = torch.Tensor(train)
    train_sum = torch.Tensor(train_sum)
    val = torch.Tensor(val)
    val_sum = torch.Tensor(val_sum)
    train_true = torch.tensor(train_true)
    val_true = torch.tensor(val_true)
    best_loss, best_acc, best_epoch, best_hidden_layer_size, best_learning_rate, best_number_of_layers, best_decay_rate = Postprocesser.neural_network(200, train, train_true, val, val_true, name)
    best_2_loss, best_2_acc, best_2_epoch, best_2_hidden_layer_size, best_2_learning_rate, best_2_number_of_layers, best_2_decay_rate = Postprocesser.neural_network(
        2, train_sum, train_true, val_sum, val_true, name + "_2")
    return val_common_acc, val_sum_acc, best_acc, best_2_acc


def postprocessor_selection():
    model = RNNWrapper.RNN(4, 200, 128, 1, True)
    model.load_state_dict(torch.load("rnn_model_state.pt"))
    model.eval()
    a1, b1, c1, d1 = run_postprocessor(model, "rnn_post_nn")
    model = LSTMWrapper.LSTM(4, 200, 50, 1, False)
    model.load_state_dict(torch.load("lstm_model_state.pt"))
    model.eval()
    a2, b2, c2, d2 = run_postprocessor(model, "lstm_post_nn")
    print(a1, b1, c1, d1)
    print(a2, b2, c2, d2)


def postprocessor_selection_age():
    model = RNNWrapper.RNN(2, 200, 128, 1, True)
    model.load_state_dict(torch.load("rnn_age_model_state.pt"))
    model.eval()
    a1, b1, c1, d1 = run_postprocessor_age(model, "rnn_age_post_nn")
    model = LSTMWrapper.LSTM(2, 200, 50, 1, True)
    model.load_state_dict(torch.load("lstm_age_model_state.pt"))
    model.eval()
    a2, b2, c2, d2 = run_postprocessor_age(model, "lstm_age_post_nn")
    print(a1, b1, c1, d1)
    print(a2, b2, c2, d2)


def generate_rnn_output(model, name, hidden_layer_size, best_number_of_layers, size):
    h = model.init_hidden(x_test.size(0))
    test_results, _ = model(x_test, h)
    post_model = FFNNWrapper.FeedForward(size, hidden_layer_size, size, best_number_of_layers)
    post_model.load_state_dict(torch.load(name + "_model_state.pt"))
    test = np.zeros([len(test_tweet_numbers), size])
    test_true = []
    j = 0
    for a, author_info in enumerate(test_tweet_numbers):
        tweet_number = author_info[0]
        test_true.append(int(author_info[int(size == 2) + 1]))
        for i in range(int(tweet_number)):
            for k, item in enumerate(test_results.__getitem__(j)):
                test[a][k] += float(item)
            j += 1
    test = torch.Tensor(test)
    test_true = torch.tensor(test_true)
    post_output = post_model(test)
    print(RNNWrapper.accuracy(post_output, test_true))
    return post_output, test_true


def comparison():
    model = RNNWrapper.RNN(4, 200, 128, 1, True)
    model.load_state_dict(torch.load("rnn_model_state.pt"))
    model.eval()
    rnn_y, y = generate_rnn_output(model, "rnn_post_nn_2", 64, 2, 4)
    model = LSTMWrapper.LSTM(4, 200, 50, 1, False)
    model.load_state_dict(torch.load("lstm_model_state.pt"))
    model.eval()
    lstm_y, _ = generate_rnn_output(model, "lstm_post_nn_2", 256, 5, 4)
    model = FFNNWrapper.FeedForward(len(Preprocesser.get_tokens()[0]) + 2, 128, 4, 1)
    model.load_state_dict(torch.load("ffnn_model_state.pt"))
    model.eval()
    ffnn_y = model(x_test_2)
    model = SVMWrapper.SVM_run(0.2, 90, 'poly', True)[1]
    svm_y = model.decision_function(x_test_2)
    print(svm_y)
    svm_y_pred = model.predict(x_test_2)
    ffnn_y_pred = np.zeros(len(ffnn_y))
    for i in range(len(ffnn_y)):
        values, indices = torch.max(ffnn_y.__getitem__(i), 0)
        ffnn_y_pred[i] = indices.item()
    rnn_y_pred = np.zeros(len(rnn_y))
    for i in range(len(rnn_y)):
        values, indices = torch.max(rnn_y.__getitem__(i), 0)
        rnn_y_pred[i] = indices.item()
    lstm_y_pred = np.zeros(len(lstm_y))
    for i in range(len(lstm_y)):
        values, indices = torch.max(lstm_y.__getitem__(i), 0)
        lstm_y_pred[i] = indices.item()
    model_2 = RNNWrapper.RNN(2, 200, 128, 1, True)
    model_2.load_state_dict(torch.load("rnn_age_model_state.pt"))
    model_2.eval()
    rnn_y_2, y_2 = generate_rnn_output(model_2, "rnn_age_post_nn_2_age", 128, 3, 2)
    model_2 = LSTMWrapper.LSTM(2, 200, 50, 1, True)
    model_2.load_state_dict(torch.load("lstm_age_model_state.pt"))
    model_2.eval()
    lstm_y_2, _ = generate_rnn_output(model_2, "lstm_age_post_nn_2_age", 512, 2, 2)
    model_2 = FFNNWrapper.FeedForward(len(Preprocesser.get_tokens()[0]) + 2, 512, 2, 2)
    model_2.load_state_dict(torch.load("ffnn_age_model_state.pt"))
    model_2.eval()
    ffnn_y_2 = model_2(x_test_2)
    model_2 = SVMWrapper.SVM_run(50000, 100, 'poly', False)[1]
    svm_y_2 = model_2.decision_function(x_test_2)
    svm_y_pred_2 = model_2.predict(x_test_2)
    ffnn_y_pred_2 = np.zeros(len(ffnn_y_2))
    for i in range(len(ffnn_y_2)):
        values, indices = torch.max(ffnn_y_2.__getitem__(i), 0)
        ffnn_y_pred_2[i] = indices.item()
    rnn_y_pred_2 = np.zeros(len(rnn_y_2))
    for i in range(len(rnn_y_2)):
        values, indices = torch.max(rnn_y_2.__getitem__(i), 0)
        rnn_y_pred_2[i] = indices.item()
    lstm_y_pred_2 = np.zeros(len(lstm_y_2))
    for i in range(len(lstm_y_2)):
        values, indices = torch.max(lstm_y_2.__getitem__(i), 0)
        lstm_y_pred_2[i] = indices.item()
    y = y.detach().numpy()
    ffnn_report = metrics.classification_report(y, ffnn_y_pred, digits=3, output_dict=True)
    print(ffnn_report)
    print(metrics.matthews_corrcoef(y, ffnn_y_pred))
    svm_report = metrics.classification_report(y, svm_y_pred, digits=3, output_dict=True)
    print(svm_report)
    print(metrics.matthews_corrcoef(y, svm_y_pred))
    rnn_report = metrics.classification_report(y, rnn_y_pred, digits=3, output_dict=True)
    print(rnn_report)
    print(metrics.matthews_corrcoef(y, rnn_y_pred))
    lstm_report = metrics.classification_report(y, lstm_y_pred, digits=3, output_dict=True)
    print(lstm_report)
    print(metrics.matthews_corrcoef(y, lstm_y_pred))
    ffnn_2_report = metrics.classification_report(y_2, ffnn_y_pred_2, digits=3, output_dict=True)
    print(ffnn_2_report)
    print(metrics.matthews_corrcoef(y_2, ffnn_y_pred_2))
    svm_2_report = metrics.classification_report(y_2, svm_y_pred_2, digits=3, output_dict=True)
    print(svm_2_report)
    print(metrics.matthews_corrcoef(y_2, svm_y_pred_2))
    rnn_2_report = metrics.classification_report(y_2, rnn_y_pred_2, digits=3, output_dict=True)
    print(rnn_2_report)
    print(metrics.matthews_corrcoef(y_2, rnn_y_pred_2))
    lstm_2_report = metrics.classification_report(y_2, lstm_y_pred_2, digits=3, output_dict=True)
    print(lstm_2_report)
    print(metrics.matthews_corrcoef(y_2, lstm_y_pred_2))
    data = [[ffnn_report["0"]["f1-score"],svm_report["0"]["f1-score"],rnn_report["0"]["f1-score"],
             lstm_report["0"]["f1-score"]],
            [ffnn_report["1"]["f1-score"], svm_report["1"]["f1-score"], rnn_report["1"]["f1-score"],
             lstm_report["1"]["f1-score"]],
            [ffnn_report["2"]["f1-score"], svm_report["2"]["f1-score"], rnn_report["2"]["f1-score"],
             lstm_report["2"]["f1-score"]],
            [ffnn_report["3"]["f1-score"], svm_report["3"]["f1-score"], rnn_report["3"]["f1-score"],
             lstm_report["3"]["f1-score"]],
            [ffnn_2_report["0"]["f1-score"], svm_2_report["0"]["f1-score"], rnn_2_report["0"]["f1-score"],
             lstm_2_report["0"]["f1-score"]],
            [ffnn_2_report["1"]["f1-score"], svm_2_report["1"]["f1-score"], rnn_2_report["1"]["f1-score"],
             lstm_2_report["1"]["f1-score"]]]
    print(wilcoxon(ffnn_y_pred, svm_y_pred, alternative="two-sided"))
    print(wilcoxon(ffnn_y_pred, rnn_y_pred, alternative="two-sided"))
    print(wilcoxon(ffnn_y_pred, lstm_y_pred, alternative="two-sided"))
    print(wilcoxon(svm_y_pred, rnn_y_pred, alternative="two-sided"))
    print(wilcoxon(svm_y_pred, lstm_y_pred, alternative="two-sided"))
    print(wilcoxon(rnn_y_pred, lstm_y_pred, alternative="two-sided"))
    print(wilcoxon(ffnn_y_pred_2, svm_y_pred_2, alternative="two-sided"))
    print(wilcoxon(ffnn_y_pred_2, rnn_y_pred_2, alternative="two-sided"))
    print(wilcoxon(ffnn_y_pred_2, lstm_y_pred_2, alternative="two-sided"))
    print(wilcoxon(svm_y_pred_2, rnn_y_pred_2, alternative="two-sided"))
    print(wilcoxon(svm_y_pred_2, lstm_y_pred_2, alternative="two-sided"))
    print(wilcoxon(rnn_y_pred_2, lstm_y_pred_2, alternative="two-sided"))
    fig = plt.figure()
    X = np.arange(4)
    ax = fig.add_subplot(111)
    ax.bar(X + 0, data[0], color='red', width=(1/7))
    ax.bar(X + (1/7), data[1], color='orange', width=(1/7))
    ax.bar(X + (2/7), data[2], color='yellow', width=(1/7))
    ax.bar(X + (3/7), data[3], color='lawngreen', width=(1/7))
    ax.bar(X + (4/7), data[4], color='blue', width=(1/7))
    ax.bar(X + (5/7), data[5], color='cyan', width=(1/7))
    plt.ylabel("F1-Score")
    plt.xlabel("Model")
    plt.xticks(X + (2.5/7), ("FFNN", "SVM", "RNN", "LSTM"))
    plt.yticks(np.arange(0, 1, 0.1))
    ax.legend(labels=["18-24", "25-35", "35-49", "50+", "Female", "Male"])
    plt.tight_layout()
    plt.show()
    plt.cla()
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    colour = dict()
    y_roc = label_binarize(y, classes=[0, 1, 2, 3])
    fpr["ffnn"], tpr["ffnn"], _ = metrics.roc_curve(y_roc.ravel(), ffnn_y.detach().numpy().ravel())
    roc_auc["ffnn"] = metrics.auc(fpr["ffnn"], tpr["ffnn"])
    colour["ffnn"] = 'darkorange'
    fpr["svm"], tpr["svm"], _ = metrics.roc_curve(y_roc.ravel(), svm_y.ravel())
    roc_auc["svm"] = metrics.auc(fpr["svm"], tpr["svm"])
    colour["svm"] = 'cornflowerblue'
    fpr["rnn"], tpr["rnn"], _ = metrics.roc_curve(y_roc.ravel(), rnn_y.detach().numpy().ravel())
    roc_auc["rnn"] = metrics.auc(fpr["rnn"], tpr["rnn"])
    colour["rnn"] = 'navy'
    fpr["lstm"], tpr["lstm"], _ = metrics.roc_curve(y_roc.ravel(), lstm_y.detach().numpy().ravel())
    roc_auc["lstm"] = metrics.auc(fpr["lstm"], tpr["lstm"])
    colour["lstm"] = 'deeppink'
    plt.figure()
    for model in fpr:
        plt.plot(fpr[model], tpr[model],
                 label='{0} micro-average ROC curve (area = {1:0.2f})'
                       ''.format(model.upper(), roc_auc[model]),
                 color=colour[model], linestyle='--', linewidth=4)
    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Age')
    plt.legend(loc="lower right")
    plt.show()
    plt.cla()
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    y_roc = label_binarize(y_2, classes=[0, 1])
    fpr["ffnn"], tpr["ffnn"], _ = metrics.roc_curve(y_roc.ravel(), ffnn_y_2.detach().numpy()[:, 1])
    roc_auc["ffnn"] = metrics.auc(fpr["ffnn"], tpr["ffnn"])
    fpr["svm"], tpr["svm"], _ = metrics.roc_curve(y_roc.ravel(), svm_y_2)
    roc_auc["svm"] = metrics.auc(fpr["svm"], tpr["svm"])
    fpr["rnn"], tpr["rnn"], _ = metrics.roc_curve(y_roc.ravel(), rnn_y_2.detach().numpy()[:, 1])
    roc_auc["rnn"] = metrics.auc(fpr["rnn"], tpr["rnn"])
    fpr["lstm"], tpr["lstm"], _ = metrics.roc_curve(y_roc.ravel(), lstm_y_2.detach().numpy()[:, 1])
    roc_auc["lstm"] = metrics.auc(fpr["lstm"], tpr["lstm"])
    plt.figure()
    for model in fpr:
        plt.plot(fpr[model], tpr[model],
             label='{0} ROC curve (area = {1:0.2f})'
                   ''.format(model.upper(), roc_auc[model]),
             color=colour[model], linestyle='--', linewidth=4)
    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Gender')
    plt.legend(loc="lower right")
    plt.show()


def generate_rnn_output_ind(model, name, hidden_layer_size, best_number_of_layers, size):
    h = model.init_hidden(x_2_social_test.size(0))
    social_test_results, _ = model(x_2_social_test, h)
    h = model.init_hidden(x_2_review_test.size(0))
    review_test_results, _ = model(x_2_review_test, h)
    h = model.init_hidden(x_2_blog_test.size(0))
    blog_test_results, _ = model(x_2_blog_test, h)
    post_model = FFNNWrapper.FeedForward(size, hidden_layer_size, size, best_number_of_layers)
    post_model.load_state_dict(torch.load(name + "_model_state.pt"))
    social_test = np.zeros([len(social_tweet_numbers), size])
    review_test = np.zeros([len(review_tweet_numbers), size])
    blog_test = np.zeros([len(blog_tweet_numbers), size])
    social_test_true = []
    j = 0
    for a, author_info in enumerate(social_tweet_numbers):
        tweet_number = author_info[0]
        social_test_true.append(int(author_info[int(size == 2) + 1]))
        for i in range(int(tweet_number)):
            if j == len(social_test_results):
                break
            for k, item in enumerate(social_test_results.__getitem__(j)):
                social_test[a][k] += float(item)
            j += 1
    social_test = torch.Tensor(social_test)
    social_test_true = torch.tensor(social_test_true)
    social_post_output = post_model(social_test)
    review_test_true = []
    j = 0
    for a, author_info in enumerate(review_tweet_numbers):
        tweet_number = author_info[0]
        review_test_true.append(int(author_info[int(size == 2) + 1]))
        for i in range(int(tweet_number)):
            for k, item in enumerate(review_test_results.__getitem__(j)):
                review_test[a][k] += float(item)
            j += 1
    review_test = torch.Tensor(review_test)
    review_test_true = torch.tensor(review_test_true)
    review_post_output = post_model(review_test)
    blog_test_true = []
    j = 0
    for a, author_info in enumerate(blog_tweet_numbers):
        tweet_number = author_info[0]
        blog_test_true.append(int(author_info[int(size == 2) + 1]))
        for i in range(int(tweet_number)):
            for k, item in enumerate(blog_test_results.__getitem__(j)):
                blog_test[a][k] += float(item)
            j += 1
    blog_test = torch.Tensor(blog_test)
    blog_test_true = torch.tensor(blog_test_true)
    blog_post_output = post_model(blog_test)

    return social_post_output, social_test_true, review_post_output, review_test_true, blog_post_output, blog_test_true


def independence_test():
    model = RNNWrapper.RNN(4, 200, 128, 1, True)
    model.load_state_dict(torch.load("rnn_model_state.pt"))
    model.eval()
    rnn_social_post_output, social_test_true, rnn_review_post_output, review_test_true, rnn_blog_post_output, blog_test_true = generate_rnn_output_ind(model, "rnn_post_nn_2", 64, 2, 4)
    model = LSTMWrapper.LSTM(4, 200, 50, 1, False)
    model.load_state_dict(torch.load("lstm_model_state.pt"))
    model.eval()
    lstm_social_post_output, _, lstm_review_post_output, _, lstm_blog_post_output, _ = generate_rnn_output_ind(model, "lstm_post_nn_2", 256, 5, 4)
    model = FFNNWrapper.FeedForward(len(Preprocesser.get_tokens()[0]) + 2, 128, 4, 1)
    model.load_state_dict(torch.load("ffnn_model_state.pt"))
    model.eval()
    ffnn_social = model(x_social_test)
    ffnn_review = model(x_review_test)
    ffnn_blog = model(x_blog_test)
    model = SVMWrapper.SVM_run(0.2, 90, 'poly', True)[1]
    svm_social = model.predict(x_social_test)
    svm_review = model.predict(x_review_test)
    svm_blog = model.predict(x_blog_test)
    ffnn_y_social_pred = np.zeros(len(ffnn_social))
    for i in range(len(ffnn_social)):
        values, indices = torch.max(ffnn_social.__getitem__(i), 0)
        ffnn_y_social_pred[i] = indices.item()
    rnn_y_social_pred = np.zeros(len(rnn_social_post_output))
    for i in range(len(rnn_social_post_output)):
        values, indices = torch.max(rnn_social_post_output.__getitem__(i), 0)
        rnn_y_social_pred[i] = indices.item()
    lstm_y_social_pred = np.zeros(len(lstm_social_post_output))
    for i in range(len(lstm_social_post_output)):
        values, indices = torch.max(lstm_social_post_output.__getitem__(i), 0)
        lstm_y_social_pred[i] = indices.item()
    ffnn_y_review_pred = np.zeros(len(ffnn_review))
    for i in range(len(ffnn_review)):
        values, indices = torch.max(ffnn_review.__getitem__(i), 0)
        ffnn_y_review_pred[i] = indices.item()
    rnn_y_review_pred = np.zeros(len(rnn_review_post_output))
    for i in range(len(rnn_review_post_output)):
        values, indices = torch.max(rnn_review_post_output.__getitem__(i), 0)
        rnn_y_review_pred[i] = indices.item()
    lstm_y_review_pred = np.zeros(len(lstm_review_post_output))
    for i in range(len(lstm_review_post_output)):
        values, indices = torch.max(lstm_review_post_output.__getitem__(i), 0)
        lstm_y_review_pred[i] = indices.item()
    ffnn_y_blog_pred = np.zeros(len(ffnn_blog))
    for i in range(len(ffnn_blog)):
        values, indices = torch.max(ffnn_blog.__getitem__(i), 0)
        ffnn_y_blog_pred[i] = indices.item()
    rnn_y_blog_pred = np.zeros(len(rnn_blog_post_output))
    for i in range(len(rnn_blog_post_output)):
        values, indices = torch.max(rnn_blog_post_output.__getitem__(i), 0)
        rnn_y_blog_pred[i] = indices.item()
    lstm_y_blog_pred = np.zeros(len(lstm_blog_post_output))
    for i in range(len(lstm_blog_post_output)):
        values, indices = torch.max(lstm_blog_post_output.__getitem__(i), 0)
        lstm_y_blog_pred[i] = indices.item()
    print(metrics.accuracy_score(social_y.detach().numpy(), ffnn_y_social_pred))
    print(metrics.accuracy_score(social_y.detach().numpy(), svm_social))
    print(metrics.accuracy_score(social_test_true, rnn_y_social_pred))
    print(metrics.accuracy_score(social_test_true, lstm_y_social_pred))
    print(metrics.accuracy_score(review_y.detach().numpy(), ffnn_y_review_pred))
    print(metrics.accuracy_score(review_y.detach().numpy(), svm_review))
    print(metrics.accuracy_score(review_test_true, rnn_y_review_pred))
    print(metrics.accuracy_score(review_test_true, lstm_y_review_pred))
    print(metrics.accuracy_score(blog_y.detach().numpy(), ffnn_y_blog_pred))
    print(metrics.accuracy_score(blog_y.detach().numpy(), svm_blog))
    print(metrics.accuracy_score(blog_test_true, rnn_y_blog_pred))
    print(metrics.accuracy_score(blog_test_true, lstm_y_blog_pred))
    model = RNNWrapper.RNN(2, 200, 128, 1, True)
    model.load_state_dict(torch.load("rnn_age_model_state.pt"))
    model.eval()
    rnn_social_post_output, social_test_true, rnn_review_post_output, review_test_true, rnn_blog_post_output, blog_test_true = generate_rnn_output_ind(model, "rnn_age_post_nn_2_age", 128, 3, 2)
    model = LSTMWrapper.LSTM(2, 200, 50, 1, True)
    model.load_state_dict(torch.load("lstm_age_model_state.pt"))
    model.eval()
    lstm_social_post_output, _, lstm_review_post_output, _, lstm_blog_post_output, _ = generate_rnn_output_ind(model, "lstm_age_post_nn_2_age", 512, 2, 2)
    model = FFNNWrapper.FeedForward(len(Preprocesser.get_tokens()[0]) + 2, 512, 2, 2)
    model.load_state_dict(torch.load("ffnn_age_model_state.pt"))
    model.eval()
    ffnn_social = model(x_social_test)
    ffnn_review = model(x_review_test)
    ffnn_blog = model(x_blog_test)
    model = SVMWrapper.SVM_run(50000, 100, 'poly', False)[1]
    svm_social = model.predict(x_social_test)
    svm_review = model.predict(x_review_test)
    svm_blog = model.predict(x_blog_test)
    ffnn_y_social_pred = np.zeros(len(ffnn_social))
    for i in range(len(ffnn_social)):
        values, indices = torch.max(ffnn_social.__getitem__(i), 0)
        ffnn_y_social_pred[i] = indices.item()
    rnn_y_social_pred = np.zeros(len(rnn_social_post_output))
    for i in range(len(rnn_social_post_output)):
        values, indices = torch.max(rnn_social_post_output.__getitem__(i), 0)
        rnn_y_social_pred[i] = indices.item()
    lstm_y_social_pred = np.zeros(len(lstm_social_post_output))
    for i in range(len(lstm_social_post_output)):
        values, indices = torch.max(lstm_social_post_output.__getitem__(i), 0)
        lstm_y_social_pred[i] = indices.item()
    ffnn_y_review_pred = np.zeros(len(ffnn_review))
    for i in range(len(ffnn_review)):
        values, indices = torch.max(ffnn_review.__getitem__(i), 0)
        ffnn_y_review_pred[i] = indices.item()
    rnn_y_review_pred = np.zeros(len(rnn_review_post_output))
    for i in range(len(rnn_review_post_output)):
        values, indices = torch.max(rnn_review_post_output.__getitem__(i), 0)
        rnn_y_review_pred[i] = indices.item()
    lstm_y_review_pred = np.zeros(len(lstm_review_post_output))
    for i in range(len(lstm_review_post_output)):
        values, indices = torch.max(lstm_review_post_output.__getitem__(i), 0)
        lstm_y_review_pred[i] = indices.item()
    ffnn_y_blog_pred = np.zeros(len(ffnn_blog))
    for i in range(len(ffnn_blog)):
        values, indices = torch.max(ffnn_blog.__getitem__(i), 0)
        ffnn_y_blog_pred[i] = indices.item()
    rnn_y_blog_pred = np.zeros(len(rnn_blog_post_output))
    for i in range(len(rnn_blog_post_output)):
        values, indices = torch.max(rnn_blog_post_output.__getitem__(i), 0)
        rnn_y_blog_pred[i] = indices.item()
    lstm_y_blog_pred = np.zeros(len(lstm_blog_post_output))
    for i in range(len(lstm_blog_post_output)):
        values, indices = torch.max(lstm_blog_post_output.__getitem__(i), 0)
        lstm_y_blog_pred[i] = indices.item()
    print(metrics.accuracy_score(social_y_2.detach().numpy(), ffnn_y_social_pred))
    print(metrics.accuracy_score(social_y_2.detach().numpy(), svm_social))
    print(metrics.accuracy_score(social_test_true, rnn_y_social_pred))
    print(metrics.accuracy_score(social_test_true, lstm_y_social_pred))
    print(metrics.accuracy_score(review_y_2.detach().numpy(), ffnn_y_review_pred))
    print(metrics.accuracy_score(review_y_2.detach().numpy(), svm_review))
    print(metrics.accuracy_score(review_test_true, rnn_y_review_pred))
    print(metrics.accuracy_score(review_test_true, lstm_y_review_pred))
    print(metrics.accuracy_score(blog_y_2.detach().numpy(), ffnn_y_blog_pred))
    print(metrics.accuracy_score(blog_y_2.detach().numpy(), svm_blog))
    print(metrics.accuracy_score(blog_test_true, rnn_y_blog_pred))
    print(metrics.accuracy_score(blog_test_true, lstm_y_blog_pred))


comparison()
