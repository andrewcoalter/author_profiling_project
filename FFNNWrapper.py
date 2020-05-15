from collections import OrderedDict
import torch
import torch.nn as nn
import Preprocesser
import matplotlib.pyplot as plt

device = torch.device('cpu' if torch.cuda.is_available() else 'cpu')
num_classes = 2


class FeedForward(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes, num_layers):
        super(FeedForward, self).__init__()
        torch.manual_seed(0)
        network = [("input", nn.Linear(input_size, hidden_size)), ("tanh0", nn.Tanh())]
        for i in range(num_layers - 1):
            network.append(("hidden_"+i.__str__(), nn.Linear(hidden_size, hidden_size)))
            network.append(("tanh"+(i+1).__str__(), nn.Tanh()))
        network.append(("output", nn.Linear(hidden_size, num_classes)))
        network.append(("logsoftmax", nn.LogSoftmax(dim=1)))
        self.net = nn.Sequential(OrderedDict(network))

    def forward(self, x):
        return self.net(x)


x_1 = []
y_1 = []
y_2 = []
y_3 = []
def fit(x, y, x_test, y_test, model, opt, loss_fn, decay_rate, name):
    best_accuracy = 0
    current_epoch = 0
    best_loss = 10
    best_epoch = 0
    epochs_exceeded = False
    model.train()
    while not epochs_exceeded:
        model.zero_grad()
        x_1.append(current_epoch)

        loss = loss_fn(model(x), y)
        y_3.append(accuracy(model(x), y))
        y_1.append(loss.item())
        test_results = model(x_test)
        current_acc = accuracy(test_results, y_test)
        current_loss = loss_fn(test_results, y_test)
        y_2.append(current_loss.item())
        if current_loss < best_loss:
            best_loss = current_loss
            best_accuracy = current_acc
            best_epoch = current_epoch
            torch.save(model.state_dict(), name + "_age_model_state.pt")
        print(current_epoch, current_acc, current_loss)
        loss.backward()
        opt.step()
        opt.zero_grad()
        if ((2 * best_epoch) < current_epoch and current_epoch > 100) or current_epoch > 1000:
            epochs_exceeded = True
        current_epoch += 1
        if current_epoch % 15 == 0:
            for param_group in opt.param_groups:
                param_group['lr'] = param_group['lr'] * (1 - decay_rate)
    return best_loss, best_accuracy, best_epoch


def accuracy(predictions, correct_indices):
    correct = 0
    number = 0
    for i in range(len(predictions)):
        values, indices = torch.max(predictions.__getitem__(i), 0)
        if indices.item() == correct_indices[i]:
            correct += 1
        number += 1
    return correct/number


def FFNN_run(hidden_size, learning_rate, num_layers, decay_rate):
    x_train = Preprocesser.fetch_author_tweets_tokens(False, False)
    x_train = x_train.to(device)
    y_train = Preprocesser.fetch_author_truths(False, False)[1]
    y_train = y_train.to(device)
    x_test = Preprocesser.fetch_author_tweets_tokens(True, True)
    x_test = x_test.to(device)
    y_test = Preprocesser.fetch_author_truths(True, True)[1]
    y_test = y_test.to(device)
    input_size = len(Preprocesser.get_tokens()[0]) + 2
    name = "ffnn"
    return FFNN_run_helper(input_size, hidden_size, learning_rate, num_layers, decay_rate, x_train, y_train, x_test, y_test, name)


def FFNN_run_helper(input_size, hidden_size, learning_rate, num_layers, decay_rate, x_train, y_train, x_test, y_test, name):

    model = FeedForward(input_size, hidden_size, num_classes, num_layers)
    model.to(device)

    loss_fn = nn.NLLLoss()
    opt = torch.optim.Adam(model.parameters(), lr=learning_rate)

    model.eval()
    print(y_test.dtype)
    y_pred = model(x_test)
    before_train = loss_fn(y_pred, y_test)
    print('Accuracy before Training', accuracy(y_pred, y_test))
    print('Test loss before Training', before_train.item())

    loss, acc, epoch = fit(x_train, y_train, x_test, y_test, model, opt, loss_fn, decay_rate, name)

    model.eval()
    y_pred = model(x_test)
    print(y_pred)
    after_train = loss_fn(y_pred, y_test)
    print('Accuracy after Training', accuracy(y_pred, y_test))
    print('Test loss after Training', after_train.item())

    return round(loss.item(), 3), round(acc, 3), epoch


