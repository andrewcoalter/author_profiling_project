import os
from html.parser import HTMLParser
import re
import numpy as np
import torch
import FFNNWrapper
import HyperParameterOptimisation


def author_accuracy(tweets_predictions, author_tweet_numbers_and_truths, gender):
    correct_max = 0
    correct_total = 0
    total = 0
    j = 0
    for author_info in author_tweet_numbers_and_truths:
        tweet_number = author_info[0]
        correct_info = author_info[1 + gender]
        predicted = np.zeros([2])
        totaling = np.zeros([2])
        for i in range(int(tweet_number)):
            for k, item in enumerate(tweets_predictions.__getitem__(j)):
                totaling[k] += item
            values, indices = torch.max(tweets_predictions.__getitem__(j), 0)
            predicted[indices.item()] += 1
            j += 1
        predicted_info = torch.Tensor(predicted)
        prediction_values, prediction_indices = torch.max(predicted_info, 0)
        if prediction_indices == correct_info:
            correct_max += 1
        totaling_info = torch.Tensor(totaling)
        totaling_values, totaling_indices = torch.max(totaling_info, 0)
        if totaling_indices == correct_info:
            correct_total += 1
        total += 1
    return correct_max/total, correct_total/total


def neural_network(input_size, x_train, y_train, x_test, y_test, name):
    best_loss = 100
    best_hidden_layer_size = 0
    best_learning_rate = 0
    best_number_of_layers = 0
    best_decay_rate = 0
    for i in range(200):
        ffnn_hidden_layer_size = HyperParameterOptimisation.generate_hidden_layer_size()
        ffnn_learning_rate = HyperParameterOptimisation.generate_learning_rate()
        ffnn_number_of_layers = HyperParameterOptimisation.generate_number_of_layers()
        ffnn_decay_rate = HyperParameterOptimisation.generate_decay_rate()
        ffnn_loss, ffnn_acc, ffnn_epoch = FFNNWrapper.FFNN_run_helper(input_size, ffnn_hidden_layer_size, ffnn_learning_rate,
                                                                ffnn_number_of_layers,
                                                                ffnn_decay_rate, x_train, y_train, x_test, y_test, name)
        HyperParameterOptimisation.write_to_file(
            [name, ffnn_loss.__str__(), ffnn_acc.__str__(), ffnn_epoch.__str__(), ffnn_hidden_layer_size.__str__(), ffnn_learning_rate.__str__(),
             ffnn_number_of_layers.__str__(), ffnn_decay_rate.__str__()])
        if ffnn_loss < best_loss:
            best_loss = ffnn_loss
            best_hidden_layer_size = ffnn_hidden_layer_size
            best_learning_rate = ffnn_learning_rate
            best_number_of_layers = ffnn_number_of_layers
            best_decay_rate = ffnn_decay_rate
    best_loss, best_acc, best_epoch = FFNNWrapper.FFNN_run_helper(input_size, best_hidden_layer_size, best_learning_rate,
                                                                  best_number_of_layers,
                                                                  best_decay_rate, x_train, y_train, x_test, y_test,
                                                                  name)
    HyperParameterOptimisation.write_to_file(
        [name, best_loss.__str__(), best_acc.__str__(), best_epoch.__str__(), best_hidden_layer_size.__str__(), best_learning_rate.__str__(),
         best_number_of_layers.__str__(), best_decay_rate.__str__()])
    print(best_loss, best_acc, best_epoch, best_hidden_layer_size, best_learning_rate, best_number_of_layers, best_decay_rate)
    return best_loss, best_acc, best_epoch, best_hidden_layer_size, best_learning_rate, best_number_of_layers, best_decay_rate