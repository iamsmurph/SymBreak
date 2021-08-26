import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import torchvision
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_curve, auc

import torchvision.transforms.functional as TF
import random

import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
# default `log_dir` is "runs" - we'll be more specific here
flag = "2000epochs90DegRotations"
writer = SummaryWriter('runs/cnn/' + flag)

class OrganoidCNNDataset(Dataset):

    def __init__(self, transform=None, train=None):
        df = np.loadtxt('scripting/dfLocalityReward.csv', delimiter=',', dtype=np.float32, skiprows=1)
        X = df[:, 1:]
        y = ((df[:, [0]] > df[:, 0].mean())*1).astype(np.float32)

        if train == True:
            self.n_samples = X[:-50].shape[0]

            self.x_data = X[:-50].reshape(-1, 1, 41, 41)
            self.y_data = y[:-50]
        else:
            self.n_samples = X[-50:].shape[0]

            self.x_data = X[-50:].reshape(-1, 1, 41, 41)
            self.y_data = y[-50:]

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
    angles = [-180, -90, 0, 90]

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
        self.fc4 = nn.Linear(32, hparam['output_size'])

    def forward(self, x):
        # -> n, 1, 41, 41
        x = self.pool(F.relu(self.conv1(x)))  # -> n, 6, 18, 18
        x = self.pool(F.relu(self.conv2(x)))  # -> n, 16, 6, 6
        x = x.view(-1, 16 * 6 * 6)            # -> n, 576
        x = F.relu(self.fc1(x))               # -> n, 128
        x = F.relu(self.fc2(x))               # -> n, 64
        x = F.relu(self.fc3(x))               # -> n, 32
        x = self.fc4(x)                       # -> n, 1
        return x

######## INITIALIZATION ########

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyper-parameters 
hparam = {"output_size": 1, "num_epochs": 2000, "batch_size": 70, "lr": 0.0001}

composed = torchvision.transforms.Compose([ToTensor(), RandomRotation()])

trainData = OrganoidCNNDataset(transform=composed, train=True)
testData = OrganoidCNNDataset(transform=composed, train=False)

train_loader = DataLoader(dataset=trainData,
                          batch_size=hparam['batch_size'],
                          shuffle=True,
                          num_workers=2)

test_loader = DataLoader(dataset=testData,
                          batch_size=hparam['batch_size'],
                          shuffle=False,
                          num_workers=2)

model = ConvNet().to(device)

dataiter = iter(train_loader)
images, labels = dataiter.next()

#writer.add_image('No rotation:', images[0], 0, dataformats='CHW')
writer.add_graph(model, images.to(device))

# Loss and optimizer
criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=hparam['lr'])  

######## Training ########
def sigmoid(x):
  return 1 / (1 + np.exp(-x))

sigmoid_v = np.vectorize(sigmoid)

running_correct = 0
running_loss = 0.0
n_total_steps = len(train_loader)

for epoch in range(hparam['num_epochs']):
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

        running_loss += loss.item()

        probs = sigmoid_v(outputs.detach().cpu().numpy())
        preds = np.where(probs > 0.5, 1, 0)
        y = labels.detach().cpu().numpy()
        running_correct += np.sum(preds == y)
        
        if (i+1) % 5 == 0: 
            print (f'Epoch [{epoch+1}/{hparam["num_epochs"]}], Step [{i+1}/{n_total_steps}], Loss: {loss.item():.4f}')
            writer.add_scalar('Training Loss', running_loss / hparam['batch_size'], epoch * n_total_steps + i)
            running_accuracy = running_correct / (hparam['batch_size']*5)
            writer.add_scalar('Training Accuracy', running_accuracy, epoch * n_total_steps + i)
            running_correct = 0
            running_loss = 0.0

######## Testing ########
def roc_auc_graph(y, probs):
    fpr, tpr, _ = roc_curve(y, probs, pos_label=1)
    lw = 2
    roc_auc = auc(fpr, tpr)
    fig = plt.figure()
    plt.plot(fpr, tpr, color='darkorange',
            lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic example')
    plt.legend(loc="lower right")
    return fig

class_labels = []
class_preds = []
# In test phase, we don't need to compute gradients (for memory efficiency)
with torch.no_grad():
    n_correct = 0
    n_samples = 0
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)
        
        probs = sigmoid_v(outputs.detach().cpu().numpy())
        preds = np.where(probs > 0.5, 1, 0)
        y = labels.detach().cpu().numpy()

        n_correct += np.sum(preds == y)
        n_samples = labels.shape[0]
        
    
    acc = 100.0 * n_correct / n_samples
    print(f'Accuracy of the network on the {n_samples} test images: {acc} %')

    #writer.add_pr_curve('test_roc_curve', labels, probs, 0)
    writer.add_scalar('Test Accuracy', acc, 0)
    writer.add_figure('Test: ROC AUC', roc_auc_graph(y, probs), 0)
    writer.close()