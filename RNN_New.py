import torch
import torch.nn as nn
import numpy as np
import Prepocesser

CATEGORY = 1

device = torch.device('cpu' if torch.cuda.is_available() else 'cpu')
x_train = Prepocesser.fetch_author_tweets_tokens_ordered(False)
x_train = x_train.to(device)
y_train = Prepocesser.fetch_author_truths(False)[0]
y_train = y_train.to(device)
x_test = Prepocesser.fetch_author_tweets_tokens_ordered(True)
x_test = x_test.to(device)
y_test = Prepocesser.fetch_author_truths(True)[0]
y_test = y_test.to(device)

tokens, longest = Prepocesser.get_tokens()
input_size = 100
hidden_size = 10
num_classes = 4
num_epochs = 10
learning_rate = 0.001


class RNN(nn.Module):
    def __init__(self, input_size, output_size, hidden_dim, n_layers):
        super(RNN, self).__init__()

        # Defining some parameters
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers

        # Defining the layers
        # RNN Layer
        self.rnn = nn.RNN(input_size, hidden_dim, n_layers, batch_first=True)
        # Fully connected layer
        self.fc = nn.Linear(hidden_dim, output_size)

    def forward(self, x):

        batch_size = x.size(0)

        # Initializing hidden state for first input using method defined below
        hidden = self.init_hidden(batch_size)

        # Passing in the input and hidden state into the model and obtaining outputs
        out, hidden = self.rnn(x, hidden)

        # Reshaping the outputs such that it can be fit into the fully connected layer
        out = out.contiguous().view(-1, self.hidden_dim)
        out = self.fc(out)

        return out, hidden

    def init_hidden(self, batch_size):
        # This method generates the first hidden state of zeros which we'll use in the forward pass
        # We'll send the tensor holding the hidden state to the device we specified earlier as well
        hidden = torch.zeros(self.n_layers, batch_size, self.hidden_dim)
        return hidden


model = RNN(input_size=input_size, output_size=num_classes, hidden_dim=hidden_size, n_layers=1)
# We'll also set the model to the device that we defined earlier (default is CPU)
model.to(device)

# Define Loss, Optimizer
loss_fn = nn.NLLLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)


def train(x, y, model, loss_fn):
    for epoch in range(num_epochs):
        print(epoch)
        optimizer.zero_grad()  # Clears existing gradients from previous epoch
        x.to(device)
        y_pred = torch.Tensor(np.zeros([len(y_train), num_classes]))
        for i in range(len(y_train)):
            y_pred_i, hidden = model(Prepocesser.fetch_author_tweets_tokens_ordered_singular(False, i))
            prob = nn.functional.softmax(y_pred_i[-1], dim=0)
            y_pred[i] = prob
        loss = loss_fn(y_pred, y_train)
        loss.backward()  # Does backpropagation and calculates gradients
        optimizer.step()  # Updates the weights accordingly


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
y_pred = np.zeros([len(y_test), num_classes])
for i in range(len(y_test)):
    y_pred_i, hidden = model(Prepocesser.fetch_author_tweets_tokens_ordered_singular(True, i))
    prob = nn.functional.softmax(y_pred_i[-1], dim=0).data
    #class_ind = torch.max(prob, dim=0)[1].item()
    y_pred[i] = prob
y_pred = torch.Tensor(y_pred)
before_train = loss_fn(y_pred, y_test)
print('Accuracy before Training', accuracy(y_pred, y_test))
print('Test loss before Training', before_train.item())

train(x_train, y_train, model, loss_fn)


model.eval()
y_pred = np.zeros([len(y_test), num_classes])
for i in range(len(y_test)):
    y_pred_i, hidden = model(Prepocesser.fetch_author_tweets_tokens_ordered_singular(True, i))
    prob = nn.functional.softmax(y_pred_i[-1], dim=0).data
    #class_ind = torch.max(prob, dim=0)[1].item()
    y_pred[i] = prob
y_pred = torch.Tensor(y_pred)
after_train = loss_fn(y_pred, y_test)
print('Accuracy after Training', accuracy(y_pred, y_test))
print('Test loss after Training', before_train.item())



