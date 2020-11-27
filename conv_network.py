import torch.nn.functional as F
import torch.optim as optim
import torch.nn as nn
from tqdm import tqdm
import numpy as np
import torch
import cv2
import os

REBUILD_DATA = False
print("Cuda avaliable:", torch.cuda.is_available())
if torch.cuda.is_available():
    device = torch.device("cuda:0")
    print("Running on GPU")
else:
    device = torch.device("cpu")
    print("Running on CPU")


class Dogs_Cats:
    IMAGE_SIZE = 50
    CATS = 'PetImages/Cat'
    DOGS = 'PetImages/Dog'
    LABELS = {CATS: 0, DOGS: 1}
    training_data = []
    cat_count = 0
    dog_count = 0

    def make_train_data(self):
        for label in self.LABELS:
            print(label)
            for f in tqdm(os.listdir(label)):
                try:
                    path = os.path.join(label, f)
                    image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
                    image = cv2.resize(image, (self.IMAGE_SIZE, self.IMAGE_SIZE))
                    self.training_data.append([np.array(image), np.eye(2)[self.LABELS[label]]])

                    if label == self.CATS:
                        self.cat_count += 1
                    elif label == self.DOGS:
                        self.dog_count += 1
                except Exception as e:
                    pass

        np.random.shuffle(self.training_data)
        np.save("training_cat_dog_data.npy", self.training_data)
        print("Cats: ", self.cat_count, ", Dogs: ", self.dog_count)


if REBUILD_DATA:
    dogs_cats_data = Dogs_Cats()
    dogs_cats_data.make_train_data()


class Neural_net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_1 = nn.Conv2d(1, 32, 5)
        self.conv_2 = nn.Conv2d(32, 64, 5)
        self.conv_3 = nn.Conv2d(64, 128, 5)
        # Use formula ((n+2p-f/s) + 1) for get the value
        self.fc1 = nn.Linear(2*2*128, 512)
        self.fc2 = nn.Linear(512, 2)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv_1(x)), (2, 2))
        x = F.max_pool2d(F.relu(self.conv_2(x)), (2, 2))
        x = F.max_pool2d(F.relu(self.conv_3(x)), (2, 2))
        x = x.view(-1, 2*2*128)  # Flatten the input
        x = F.relu(self.fc1(x))
        return F.softmax(self.fc2(x), dim=1)


Conv_net = Neural_net().to(device)
training_data = np.load('training_cat_dog_data.npy', allow_pickle=True)

X = torch.Tensor([i[0] for i in training_data]).view(-1, 50, 50)
X = X/255.0
y = torch.Tensor([i[1] for i in training_data])

VAL_PERCENT = 0.1
val_size = int(len(X)*VAL_PERCENT)

train_X = X[:-val_size]
train_Y = y[:-val_size]

test_X = X[-val_size:]
test_Y = y[-val_size:]

BATCH_SIZE = 100
EPOCHS = 10


def train(net):
    optimizer = optim.Adam(Conv_net.parameters(), lr=0.001)
    loss_function = nn.MSELoss()
    for epoch in range(EPOCHS):
        for i in tqdm(range(0, len(train_X), BATCH_SIZE)):
            batch_X = train_X[i:i + BATCH_SIZE].view(-1, 1, 50, 50).to(device)
            batch_Y = train_Y[i:i + BATCH_SIZE].to(device)

            net.zero_grad()
            outputs = net(batch_X)
            loss = loss_function(outputs, batch_Y)
            loss.backward()
            optimizer.step()
        print(f"Epochs: {epoch}, Loss: {loss}")


def test(net):
    correct = 0
    total = 0
    with torch.no_grad():
        for i in tqdm(range(len(test_X))):
            real_class = torch.argmax(test_Y[i]).to(device)
            net_out = net(test_X[i].view(-1, 1, 50, 50).to(device))[0]
            pred_class = torch.argmax(net_out)

            if pred_class == real_class:
                correct += 1
            total += 1

    print('Accuracy: ', round(correct / total, 3))


train(Conv_net)
test(Conv_net)
