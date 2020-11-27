from torchvision import datasets, transforms
import torch.optim as optim
import torch.nn.functional as F
import torch.nn as nn
import torchvision
import torch

# Import MNIST data
train = datasets.MNIST("", train=True, download=True, transform=transforms.Compose([transforms.ToTensor()]))
test = datasets.MNIST("", train=False, download=True, transform=transforms.Compose([transforms.ToTensor()]))

# After import convert the data in dataset object with batch
train_set = torch.utils.data.DataLoader(train, batch_size=10, shuffle=True)
test_set = torch.utils.data.DataLoader(test, batch_size=10, shuffle=True)


# Building neural network using pytorch
class Neural_Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear((28*28), 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 64)
        self.fc4 = nn.Linear(64, 10)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return F.log_softmax(x, dim=1)


linear_net = Neural_Net()
optimizer = optim.Adam(linear_net.parameters(), lr=0.001)
EPOCHS = 2

for epoch in range(EPOCHS):
    for data in train_set:
        X, y = data
        linear_net.zero_grad()
        output = linear_net(X.view(-1, 28*28))
        loss = F.nll_loss(output, y)
        loss.backward()
        optimizer.step()
    print(loss)

