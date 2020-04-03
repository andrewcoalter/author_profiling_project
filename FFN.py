import torch
import torch.nn as nn
import Prepocesser

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
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
num_classes = 4
num_epochs = 1000
learning_rate = 0.1


# Fully connected feed forward neural network with one hidden layer
class FeedForward(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(FeedForward, self).__init__()
        torch.manual_seed(0)
        self.net = nn.Sequential(  # sequential operation
            nn.Linear(input_size, hidden_size),
            nn.Sigmoid(),
            nn.Linear(hidden_size, num_classes),
            nn.Softmax())

    def forward(self, x):
        return self.net(x)


model = FeedForward(input_size, hidden_size, num_classes)
model.to(device)

def fit(x, y, model, opt, loss_fn):
    for epoch in range(num_epochs):
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
opt = torch.optim.SGD(model.parameters(), lr=learning_rate)

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



