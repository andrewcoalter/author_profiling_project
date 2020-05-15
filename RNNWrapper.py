import torch
import torch.backends.cudnn as cudnn
import torch.backends.cudnn.rnn
import torch.nn as nn
import numpy as np
import Preprocesser
import Postprocesser

torch.backends.cudnn.enabled = True
device = torch.device('cpu' if torch.cuda.is_available() else 'cpu')
x_train, _, y_train, train_tweet_numbers = Preprocesser.fetch_tweets_tokens_ordered_with_truths(False, False)
x_train = x_train.to(device)
y_train = y_train.to(device)
x_test, _, y_test, test_tweet_numbers = Preprocesser.fetch_tweets_tokens_ordered_with_truths(True, True)
x_test = x_test.to(device)
y_test = y_test.to(device)

input_size = 200
num_classes = 2


def author_accuracy(tweets_predictions, author_tweet_numbers_and_truths, gender):
    correct_max = 0
    correct_total = 0
    total = 0
    j = 0
    for author_info in author_tweet_numbers_and_truths:
        tweet_number = author_info[0]
        correct_info = author_info[1 + gender]
        predicted = np.zeros([num_classes])
        totaling = np.zeros([num_classes])
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


class RNN(nn.Module):
    def __init__(self, output_size, embedding_dim, hidden_dim, n_layers, bidirectional, drop_prob=0.5):
        super(RNN, self).__init__()
        torch.manual_seed(0)
        self.output_size = output_size
        self.n_layers = n_layers
        self.hidden_dim = hidden_dim
        self.bidirectional = bidirectional

        self.rnn = nn.RNN(embedding_dim, hidden_dim, n_layers, dropout=drop_prob, batch_first=True, bidirectional=bidirectional)
        print(self.rnn.modules())
        self.dropout = nn.Dropout(drop_prob)
        self.fc = nn.Linear((self.bidirectional + 1) * hidden_dim, output_size)
        self.logsoftmax= nn.LogSoftmax(dim=1)

    def forward(self, x, hidden):
        batch_size = x.size(0)
        rnn_out, hidden = self.rnn(x, hidden)
        rnn_out = rnn_out.contiguous().view(-1, (self.bidirectional + 1) * self.hidden_dim)

        out = self.dropout(rnn_out)
        out = self.fc(out)
        out = self.logsoftmax(out)

        out = out.view(batch_size, -1)
        out = out[:, -self.output_size:]

        return out, hidden 

    def init_hidden(self, batch_size):
        weight = next(self.parameters()).data
        hidden = weight.new((self.bidirectional + 1) * self.n_layers, batch_size, self.hidden_dim).zero_().to(device)
        return hidden


def fit(x, y, model, opt, loss_fn, batch_size, decay_rate):
    best_loss = 10
    best_epoch = 0
    best_author_accuracy = 0
    current_epoch = 0
    epochs_exceeded = False
    model.train()
    while not epochs_exceeded:
        print(current_epoch)
        model.zero_grad()
        batches = torch.split(x, batch_size, dim=0)
        y_pred = torch.Tensor(np.zeros([x.size(0), num_classes]))
        y_pred = y_pred.to(device)
        for i, batch in enumerate(batches):
            h = model.init_hidden(batch.size(0))
            outputs, h = model(batch, h)
            for j, output in enumerate(outputs):
                y_pred[i * batch_size + j] = output
        loss = loss_fn(y_pred, y)
        h = model.init_hidden(x_test.size(0))
        test_results, _ = model(x_test, h)
        current_loss = loss_fn(test_results, y_test)
        current_author_accuracy1, current_author_accuracy2 = author_accuracy(test_results, test_tweet_numbers, 1)
        print(current_loss, best_loss)
        if current_loss < best_loss:
            best_loss = current_loss
            best_author_accuracy = current_author_accuracy2
            best_epoch = current_epoch
            print("SAVE")
            torch.save(model.state_dict(), "rnn_age_model_state.pt")
        print(accuracy(y_pred, y), accuracy(test_results, y_test), current_author_accuracy1, current_author_accuracy2, loss, current_loss)
        loss.backward()
        opt.step()
        opt.zero_grad()
        if ((2 * best_epoch) < current_epoch and current_epoch > 100) or current_epoch > 1000:
            epochs_exceeded = True
        current_epoch += 1
        if current_epoch % 15 == 0:
            for param_group in opt.param_groups:
                param_group['lr'] = param_group['lr'] * (1 - decay_rate)
    return best_loss, best_author_accuracy, best_epoch


def accuracy(predictions, correct_indices):
    correct = 0
    number = 0
    for i in range(len(predictions)):
        values, indices = torch.max(predictions.__getitem__(i), 0)
        if indices.item() == correct_indices[i]:
            correct += 1
        number += 1
    return correct/number


def RNN_run(hidden_size, learning_rate, batch_size, num_layers, bidirectonial, decay_rate):

    model = RNN(num_classes, input_size, hidden_size, num_layers, bidirectonial)
    model.to(device)

    loss_fn = nn.NLLLoss()
    opt = torch.optim.Adam(model.parameters(), lr=learning_rate)

    model.eval()
    h = model.init_hidden(x_test.size(0))
    y_pred, h = model(x_test, h)
    before_train = loss_fn(y_pred, y_test)
    print('Accuracy before Training', accuracy(y_pred, y_test))
    print('Test loss before Training', before_train.item())

    loss, auth_acc, epoch = fit(x_train, y_train, model, opt, loss_fn, batch_size, decay_rate)

    model.eval()
    h = model.init_hidden(x_test.size(0))
    y_pred, h = model(x_test, h)
    print(y_pred)
    after_train = loss_fn(y_pred, y_test)
    print('Accuracy after Training', accuracy(y_pred, y_test))
    print('Test loss after Training', after_train.item())

    return loss.__str__(), auth_acc.__str__(), epoch.__str__()


#(RNN_run(128, 0.004, 10, 1, True, 0))

