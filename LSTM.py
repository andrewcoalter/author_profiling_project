import torch
import torch.nn as nn
import numpy as np
import Prepocesser

device = torch.device('cpu' if torch.cuda.is_available() else 'cpu')
x_train = Prepocesser.fetch_author_tweets_tokens(False)
x_train = x_train.to(device)
y_train = Prepocesser.fetch_author_truths(False)[1]
y_train = y_train.to(device)
x_test = Prepocesser.fetch_author_tweets_tokens(True)
x_test = x_test.to(device)
y_test = Prepocesser.fetch_author_truths(True)[1]
y_test = y_test.to(device)

input_size = len(Prepocesser.get_tokens())
hidden_size = 300
num_classes = 2
num_epochs = 1000
learning_rate = 0.005


class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size
        self.i2h = nn.LSTM(input_size + hidden_size, hidden_size)
        self.i2o = nn.LSTM(input_size + hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=0)

    def forward(self, input, hidden):
        combined = torch.cat([input, hidden], dim=0)
        combined.to(device)
        hidden = self.i2h(combined)
        output = self.i2o(combined)
        output = self.softmax(output)
        return output, hidden

    def initHidden(self):
        return torch.zeros(self.hidden_size)


model = RNN(input_size, hidden_size, num_classes)
model.to(device)

loss_fn = nn.NLLLoss()


def train(x, y, model, loss_fn):
    for epoch in range(num_epochs):
        print(epoch)
        hidden = model.initHidden()
        hidden.to(device)

        model.zero_grad()

        outputs = torch.zeros([x.size()[0], num_classes])
        for i in range(x.size()[0]):
            output, hidden = model(x[i], hidden)
            outputs[i] = output

        loss = loss_fn(outputs, y)
        loss.backward()

        for p in model.parameters():
            p.data.add_(-learning_rate, p.grad.data)

    return loss.item()


def accuracy(predictions, correct_indices):
    correct = 0
    number = 0
    for i in range(len(predictions)):
        values, indices = torch.max(predictions.__getitem__(i), 0)
        if indices.item() == correct_indices[i]:
            correct += 1
        number += 1
    return correct/number


model.eval()
hidden = model.initHidden()
hidden.to(device)
y_pred = torch.zeros([x_test.size()[0], num_classes])
y_pred.to(device)
for i, x in enumerate(x_test):
    x.to(device)
    y_pred[i] = (model(x, hidden)[0].to(device))
before_train = loss_fn(y_pred, y_test)
print('Accuracy before Training', accuracy(y_pred, y_test))
print('Test loss before Training', before_train.item())

train(x_train, y_train, model, loss_fn)

model.eval()
hidden = model.initHidden()
hidden.to(device)
y_pred = torch.zeros([x_test.size()[0], num_classes])
y_pred.to(device)
for i, x in enumerate(x_test):
    x.to(device)
    y_pred[i] = (model(x, hidden)[0].to(device))
print(y_pred)
after_train = loss_fn(y_pred, y_test)
print('Accuracy after Training', accuracy(y_pred, y_test))
print('Test loss after Training', after_train.item())



