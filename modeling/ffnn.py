import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import matplotlib


class OrganoidFFNDataset(Dataset):

    def __init__(self, transform=None, train=None):
        df = np.loadtxt('scripting/dfLocalityReward.csv', delimiter=',', dtype=np.float32, skiprows=1)
        
        if train == True:
            self.n_samples = df[:-50, :].shape[0]

            self.x_data = df[:-50, 1:]
            self.y_data = df[:-50, [0]]
        else:
            self.n_samples = df[-50:, :].shape[0]

            self.x_data = df[-50:, 1:]
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

# Fully connected neural network with one hidden layer
class NeuralNet(nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.input_size = input_size
        self.relu = nn.ReLU()
        self.l1 = nn.Linear(input_size, 128)
        self.l2 = nn.Linear(128, 64)
        self.l3 = nn.Linear(64, 32)
        self.l4 = nn.Linear(32, output_size)
    
    def forward(self, x):
        out = self.l1(x)
        out = self.relu(out)
        out = self.l2(out)
        out = self.relu(out)
        out = self.l3(out)
        out = self.relu(out)
        out = self.l4(out)
        return out

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyper-parameters 
input_size = 1681 # 41x41
output_size = 1
num_epochs = 1000
batch_size = 70
learning_rate = 0.0001

trainData = OrganoidFFNDataset(transform=ToTensor(), train=True)
testData = OrganoidFFNDataset(transform=ToTensor(), train=False)

train_loader = DataLoader(dataset=trainData,
                          batch_size=batch_size,
                          shuffle=True,
                          num_workers=2)

test_loader = DataLoader(dataset=testData,
                          batch_size=batch_size,
                          shuffle=False,
                          num_workers=2)

model = NeuralNet(input_size, output_size).to(device)

# Loss and optimizer
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)  

# Train the model
n_total_steps = len(train_loader)
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):  
        
        images = images.to(device) #.reshape(-1, input_size) if necessary
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
plt.savefig("FFNNPreds.png")
plt.show()
