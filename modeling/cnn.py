import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

import torchvision.transforms.functional as TF
import random

class OrganoidCNNDataset(Dataset):

    def __init__(self, transform=None, train=None):
        df = np.loadtxt('scripting/dfLocalityReward.csv', delimiter=',', dtype=np.float32, skiprows=1)
        
        if train == True:
            self.n_samples = df[:-50, :].shape[0]

            self.x_data = df[:-50, 1:].reshape(-1, 1, 41, 41)
            self.y_data = df[:-50, [0]]
        else:
            self.n_samples = df[-50:, :].shape[0]

            self.x_data = df[-50:, 1:].reshape(-1, 1, 41, 41)
            self.y_data = df[-50:, [0]]

        self.transform = transform

    def __getitem__(self, index):
        sample = self.x_data[index], self.y_data[index]

        if self.transform:
            sample = self.transform(sample)

        return sample

    def __len__(self):
        return self.n_samples

class ToTensor:
    # Convert ndarrays to Tensors
    def __call__(self, sample):
        inputs, targets = sample
        return torch.from_numpy(inputs), torch.from_numpy(targets)

class RandomRotation:
    #Rotate by one of the given angles

    #def __init__(self, angles
    angles = [-180, -90, -60, -30, 0, 30, 60, 90, 180]

    def __call__(self, sample):
        angle = random.choice(self.angles)
        return TF.rotate(sample[0], angle), sample[1]

class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 6)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 6)
        self.fc1 = nn.Linear(16 * 6 * 6, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 32)
        self.fc4 = nn.Linear(32, output_size)

    def forward(self, x):
        # -> n, 1, 41, 41
        x = self.pool(F.relu(self.conv1(x)))  # -> n, 6, 18, 18
        x = self.pool(F.relu(self.conv2(x)))  # -> n, 16, 6, 6
        x = x.view(-1, 16 * 6 * 6)            # -> n, 576
        x = F.relu(self.fc1(x))               # -> n, 128
        x = F.relu(self.fc2(x))               # -> n, 64
        x = F.relu(self.fc3(x))               # -> n, 32
        x = F.relu(self.fc4(x))               # -> n, 1
        return x

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyper-parameters 
input_size = 1681 # 41x41
output_size = 1
num_epochs = 10000
batch_size = 70
learning_rate = 0.001

composed = torchvision.transforms.Compose([ToTensor(), RandomRotation()])

trainData = OrganoidCNNDataset(transform=composed, train=True)
testData = OrganoidCNNDataset(transform=composed, train=False)

train_loader = DataLoader(dataset=trainData,
                          batch_size=batch_size,
                          shuffle=True,
                          num_workers=2)
'''
test_loader = DataLoader(dataset=testData,
                          batch_size=batch_size,
                          shuffle=False,
                          num_workers=2)
'''

model = ConvNet().to(device)

# Loss and optimizer
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)  

n_total_steps = len(train_loader)
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        
        images = images.to(device)
        labels = labels.to(device)

        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (i+1) % 5 == 0:
            print (f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{n_total_steps}], Loss: {loss.item():.4f}')

# In test phase, we don't need to compute gradients (for memory efficiency)
with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(device) # .reshape(-1, input_size) if necessary
        labels = labels.to(device)
        outputs = model(images)

df = np.loadtxt('scripting/dfLocalityReward.csv', delimiter=',', dtype=np.float32, skiprows=1)
preds = outputs.detach().cpu().numpy()
y = df[-50:, [0]]

#print(np.array(list(zip(preds, y))))

newx = list(range(50))
plt.scatter(newx, preds)
plt.scatter(newx, y, c = 'red')
plt.savefig("CNNPreds.png")
plt.show()