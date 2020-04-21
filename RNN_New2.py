import torch
import torch.nn as nn
import numpy as np
import Prepocesser

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
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
num_epochs = 500
learning_rate = 0.008


# Fully connected feed forward neural network with one hidden layer
class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(RNN, self).__init__()
        torch.manual_seed(0)
        self.hidden_dim = hidden_size
        self.rnn = nn.RNN(input_size, hidden_size, 1, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        batch_size = x.size(0)

        # Initializing hidden state for first input using method defined below
        hidden = self.init_hidden(batch_size)

        # Passing in the input and hidden state into the model and obtaining outputs
        out, hidden = self.rnn(x, hidden)
        print(out.size())

        # Reshaping the outputs such that it can be fit into the fully connected layer
        filter = torch.Tensor(np.zeros([out.size(0), out.size(2)]))
        filter = filter.to(device)
        for i, batch in enumerate(out):
            filter[i] = out[i][out.size(1) - 1]
        out = self.fc(filter)
        out = nn.functional.log_softmax(out, dim=1)
        return out

    def init_hidden(self, batch_size):
        hidden = torch.zeros(1, batch_size, self.hidden_dim)
        return hidden.to(device)


model = RNN(input_size, hidden_size, num_classes)
model.to(device)


def fit(x, y, model, opt, loss_fn):
    model.train()
    for epoch in range(num_epochs):
        print(epoch)
        model.zero_grad()

        loss = loss_fn(model(x), y)

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
y_pred = model(x_test)
before_train = loss_fn(y_pred, y_test)
print('Accuracy before Training', accuracy(y_pred, y_test))
print('Test loss before Training', before_train.item())

fit(x_train, y_train, model, opt, loss_fn)

model.eval()
y_pred = model(x_test)
print(y_pred)
after_train = loss_fn(y_pred, y_test)
print('Accuracy after Training', accuracy(y_pred, y_test))
print('Test loss after Training', after_train.item())


