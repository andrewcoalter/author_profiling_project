import LSTMWrapper as LSTM
import RNNWrapper as RNN
import FFNNWrapper
import SVMWrapper
import random
import numpy as np
import os


def generate_hidden_layer_size():
    return int(1 * np.float_power(2, random.randint(5, 10)))


def generate_rnn_hidden_layer_size():
    return int(1 * np.float_power(2, random.randint(5, 8)))


def generate_learning_rate():
    return random.randint(1, 10) * np.float_power(10, random.randint(-3, 1))


def generate_number_of_layers():
    return int(random.randint(1, 5))


def generate_batch_size():
    return int(random.randint(1, 10) * 10)


def generate_bidirectionality():
    return bool(random.randint(0, 1))


def generate_decay_rate():
    return 1 * np.float_power(2, random.randint(-6, -2))


def generate_c():
    return random.randint(1, 10) * np.float_power(10, random.randint(-6, 6))


def generate_gamma():
    return random.randint(1, 10) * np.float_power(10, random.randint(-6, 6))


def generate_kernel():
    return ["linear", "poly", "rbf", "sigmoid"][random.randint(0, 3)]


def write_to_file(list):
    filename = "hyperparameter_optimisation.txt"
    if os.path.exists(filename):
        append_write = "a"
    else:
        append_write = "w"
    print("append_write")
    file = open(filename, append_write)
    print("file")
    line = ""
    first = True
    for item in list:
        if not first:
            line += " & "
        else:
            first = False
        line += item
    print(line)
    print(file)
    line += "\n"
    file.write(line)
    file.close()


def run_ffnn():
    ffnn_hidden_layer_size = generate_hidden_layer_size()
    ffnn_learning_rate = generate_learning_rate()
    ffnn_number_of_layers = generate_number_of_layers()
    ffnn_decay_rate = generate_decay_rate()
    print(ffnn_hidden_layer_size, ffnn_learning_rate, ffnn_number_of_layers,
                                                               ffnn_decay_rate)
    ffnn_loss, ffnn_acc, ffnn_epoch = FFNNWrapper.FFNN_run(ffnn_hidden_layer_size, ffnn_learning_rate, ffnn_number_of_layers,
                                                               ffnn_decay_rate)
    write_to_file(["FFNN", ffnn_loss, ffnn_acc, ffnn_epoch, ffnn_hidden_layer_size.__str__(), ffnn_learning_rate.__str__(), ffnn_number_of_layers.__str__(), ffnn_decay_rate.__str__()])


def run_svm():
    svm_c = generate_c()
    svm_gamma = generate_gamma()
    svm_kernel = generate_kernel()
    try:
        svm_acc = SVMWrapper.SVM_run(svm_c, svm_gamma, svm_kernel)[0]
    except:
        svm_acc = "Exception"
    write_to_file(["SVM", svm_acc, svm_c.__str__(), svm_gamma.__str__(), svm_kernel])


def run_rnn():
    rnn_hidden_layer_size = generate_rnn_hidden_layer_size()
    rnn_learning_rate = generate_learning_rate()
    rnn_batch_size = generate_batch_size()
    rnn_number_of_layers = generate_number_of_layers()
    rnn_bidirectionality = generate_bidirectionality()
    rnn_decay_rate = generate_decay_rate()
    try:
        rnn_loss, rnn_acc, rnn_epoch = RNN.RNN_run(rnn_hidden_layer_size, rnn_learning_rate, rnn_batch_size, rnn_number_of_layers, rnn_bidirectionality, rnn_decay_rate)
    except:
        rnn_loss, rnn_acc, rnn_epoch = "Exception", "Exception", "Exception"
    write_to_file(["RNN", rnn_loss, rnn_acc, rnn_epoch, rnn_hidden_layer_size.__str__(), rnn_learning_rate.__str__(), rnn_batch_size.__str__(), rnn_number_of_layers.__str__(), rnn_bidirectionality.__str__(), rnn_decay_rate.__str__()])


def run_lstm():
    lstm_hidden_layer_size = generate_rnn_hidden_layer_size()
    lstm_learning_rate = generate_learning_rate()
    lstm_batch_size = generate_batch_size()
    lstm_number_of_layers = generate_number_of_layers()
    lstm_bidirectionality = generate_bidirectionality()
    lstm_decay_rate = generate_decay_rate()
    try:
        lstm_loss, lstm_acc, lstm_epoch = LSTM.LSTM_run(lstm_hidden_layer_size, lstm_learning_rate, lstm_batch_size, lstm_number_of_layers, lstm_bidirectionality, lstm_decay_rate)
    except:
        lstm_loss, lstm_acc, lstm_epoch = "Exception", "Exception", "Exception"
    write_to_file(["LSTM", lstm_loss, lstm_acc, lstm_epoch, lstm_hidden_layer_size.__str__(), lstm_learning_rate.__str__(), lstm_batch_size.__str__(), lstm_number_of_layers.__str__(), lstm_bidirectionality.__str__(), lstm_decay_rate.__str__()])


optimising_number = 1000
for i in range(optimising_number):
    run_ffnn()
    run_svm()
    run_rnn()
    run_lstm()