import torch
import torch.nn as nn
import numpy as np
import Prepocesser

device = torch.device('cpu' if torch.cuda.is_available() else 'cpu')
x_train = Prepocesser.fetch_author_tweets_tokens_ordered(False)
x_train = x_train.to(device)
y_train = Prepocesser.fetch_author_truths(False)[0]
y_train = y_train.to(device)
x_test = Prepocesser.fetch_author_tweets_tokens_ordered(True)
x_test = x_test.to(device)
y_test = Prepocesser.fetch_author_truths(True)[0]
y_test = y_test.to(device)

input_size = 200
hidden_size = 50
num_classes = 4
num_epochs = 100
learning_rate = 0.01


class LSTM(nn.Module):
    def __init__(self, output_size, embedding_dim, hidden_dim, n_layers, drop_prob=0.5):
        super(LSTM, self).__init__()
        self.output_size = output_size
        self.n_layers = n_layers
        self.hidden_dim = hidden_dim

        self.lstm = nn.LSTM(embedding_dim, hidden_dim, n_layers, dropout=drop_prob, batch_first=True)
        self.dropout = nn.Dropout(drop_prob)
        self.fc = nn.Linear(hidden_dim, output_size)
        self.sigmoid = nn.LogSoftmax(dim=1)

    def forward(self, x, hidden):
        batch_size = x.size(0)
        lstm_out, hidden = self.lstm(x, hidden)
        lstm_out = lstm_out.contiguous().view(-1, self.hidden_dim)

        out = self.dropout(lstm_out)
        out = self.fc(out)
        out = self.sigmoid(out)

        out = out.view(batch_size, -1)
        out = out[:, -self.output_size:]

        return out

    def init_hidden(self, batch_size):
        weight = next(self.parameters()).data
        hidden = (weight.new(self.n_layers, batch_size, self.hidden_dim).zero_().to(device),
                  weight.new(self.n_layers, batch_size, self.hidden_dim).zero_().to(device))
        return hidden


model = LSTM(num_classes, input_size, hidden_size, 2)
model.to(device)

loss_fn = nn.NLLLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)


def fit(x, y, model, opt, loss_fn):
    model.train()
    for epoch in range(num_epochs):
        h = model.init_hidden(x.size(0))
        h = tuple([e.data for e in h])
        print(epoch)
        model.zero_grad()

        loss = loss_fn(model(x, h), y)

        loss.backward()
        opt.step()
        opt.zero_grad()

    return loss.item()


def accuracy(predictions, correct_indices):
    correct = 0
    number = 0
    for i in range(len(predictions)):
        values, indices = torch.max(predictions.__getitem__(i), 0)
        print(values, indices)
        if indices.item() == correct_indices[i]:
            correct += 1
        number += 1
    return correct/number


loss_fn = nn.CrossEntropyLoss()
opt = torch.optim.Adam(model.parameters(), lr=learning_rate)

model.eval()
h = model.init_hidden(x_test.size(0))
h = tuple([e.data for e in h])
y_pred = model(x_test, h)
print(y_pred.squeeze().size())
before_train = loss_fn(y_pred, y_test)
print('Accuracy before Training', accuracy(y_pred, y_test))
print('Test loss before Training', before_train.item())

fit(x_train, y_train, model, opt, loss_fn)

model.eval()
h = model.init_hidden(x_test.size(0))
h = tuple([e.data for e in h])
y_pred = model(x_test, h)
print(y_pred)
after_train = loss_fn(y_pred, y_test)
print('Accuracy after Training', accuracy(y_pred, y_test))
print('Test loss after Training', after_train.item())



