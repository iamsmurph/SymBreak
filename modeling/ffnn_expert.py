import torch
from torch.utils.data import Dataset, DataLoader, random_split
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
import random
import torchvision.transforms.functional as TF
import torchvision
from torch.optim.lr_scheduler import ExponentialLR
import os
from sklearn.model_selection import train_test_split


from torch.utils.tensorboard import SummaryWriter
flag = "1000epochs_dropout_2hidden_stratified_ExpertFeat_Dipole_3478"
writer = SummaryWriter('modeling/runs/ffnn/expertFeatRandomVer1exp3478/' + flag)

class OrganoidFFNDataset(Dataset):

    def __init__(self, X, y):
        self.n_samples = X.shape[0]
        self.x_data = torch.from_numpy(X)
        self.y_data = torch.from_numpy(y)

    def __getitem__(self, index):
        sample = self.x_data[index], self.y_data[index]

        return sample

    def __len__(self):
        return self.n_samples

class ToTensor:
    # Convert ndarrays to Tensors
    def __call__(self, sample):
        inputs, targets = sample
        inputs, targets = torch.from_numpy(inputs), torch.from_numpy(targets)
        targets = targets.to(torch.float32)
        return inputs, targets

class RandomRotation:
    #Rotate by one of the given angles
    angles = [-180, -90, 0, 90]

    def __call__(self, sample):
        angle = random.choice(self.angles)
        sampRotate = TF.rotate(sample[0].reshape(-1, 1, 11, 11), angle)
        sampFlatten = torch.flatten(sampRotate)
        # sampRotate.reshape(41*41)
        return sampFlatten, sample[1]

# Fully connected neural network with one hidden layer
class NeuralNet(nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.input_size = input_size
        self.relu = nn.ReLU()


        self.l1 = nn.Linear(input_size, 10, bias=True)
        self.l2 = nn.Linear(10, 10, bias=True)
        self.lf = nn.Linear(10, 1, bias=True)
        self.dropout = nn.Dropout(0.25)
    
    def forward(self, x):
        out = self.l1(x)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.l2(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.lf(out)
        #out = self.dropout(out)
        return out

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyper-parameters 
hparam = {"input_size": 3, "output_size": 1, "num_epochs": 1000, 
            "batch_size": 60, "lr": 1e-4, "lrDecay": 0.995, "weight_decay": 1e-6,
            "valProp": .2, "trShuffle": True, "tstShuffle": False}

#composed = torchvision.transforms.Compose([RandomRotation()]) 
df = np.load("datasets/df_expfeat_Ver1_3478.npy")
    
X = df[:, -3:].astype(np.float32)
binaryTargs = (df[:, -4] > 0.125)*1
y = binaryTargs.astype(np.float32).reshape(-1, 1)
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=hparam["valProp"], stratify=y)

n = X_val.shape[0]
posTargs = np.sum(X_val)
weight = (n-posTargs)/posTargs

print(np.sum(y_train))
print(np.sum(y_val))

trainDataset = OrganoidFFNDataset(X_train, y_train)
valDataset = OrganoidFFNDataset(X_val, y_val)
nTrain = len(trainDataset)

#dataset = OrganoidFFNDataset()
#dataSize = len(dataset)
#nVal = int(dataSize * hparam['valProp'])
#nTrain = dataSize - nVal
#train_dataset, val_dataset = random_split(dataset, (nTrain, nVal))

#trainData = OrganoidFFNDataset(transform = composed, train=True)
#testData = OrganoidFFNDataset(transform = ToTensor(), train=False)

train_loader = DataLoader(dataset = trainDataset,
                          batch_size = hparam["batch_size"],
                          shuffle = hparam['trShuffle'],
                          num_workers = 2)

val_loader = DataLoader(dataset = valDataset,
                          batch_size = hparam["batch_size"],
                          shuffle = hparam['tstShuffle'],
                          num_workers = 2)

model = NeuralNet(hparam['input_size'], hparam["output_size"]).to(device) 

# save graph
examples = iter(val_loader)
example_data, _ = examples.next()
dataiter = iter(train_loader)
images, labels = dataiter.next()
writer.add_graph(model, example_data.to(device))

# Loss and optimizer
pos_weight = torch.ones([1]) * weight
criterion = nn.BCEWithLogitsLoss(pos_weight= pos_weight.to(device))
optimizer = torch.optim.Adam(model.parameters(), lr = hparam['lr'])  #weight_decay = hparam["weight_decay"]
#scheduler = ExponentialLR(optimizer, gamma = hparam['lrDecay'])

######## Training ########
def sigmoid(x):
  return 1 / (1 + np.exp(-x))

sigmoid_v = np.vectorize(sigmoid)

# Train the model
running_correct = 0
running_count = 0
n_total_steps = len(train_loader)
nSteps = nTrain // hparam["batch_size"]
for epoch in range(hparam['num_epochs']):
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

        probs = sigmoid_v(outputs.detach().cpu().numpy())
        preds = np.where(probs > 0.5, 1, 0)
        y = labels.detach().cpu().numpy()

        running_correct += np.sum(preds == y)
        running_count += y.shape[0]
        
        if (i + 1) % n_total_steps == 0:
            print(f'Epoch [{epoch+1}/{hparam["num_epochs"]}], Step [{i+1}/{n_total_steps}], Loss: {loss.item():.4f}')
            writer.add_scalar('Training Accuracy', np.sum(preds == y) / len(y), epoch * n_total_steps)
            writer.add_scalar('Training Loss', loss.item(), epoch * n_total_steps)

    #scheduler.step()

######## Testing ########
def roc_auc_graph(y, probs):
    fpr, tpr, _ = roc_curve(y, probs, pos_label=1)
    lw = 2
    roc_auc = auc(fpr, tpr)
    fig = plt.figure(figsize=(10, 10))
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
    posCnt = 0

    yCache = []
    probsCache = []
    for images, labels in val_loader:
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)
        
        probs = sigmoid_v(outputs.detach().cpu().numpy())
        preds = np.where(probs > 0.5, 1, 0)
        y = labels.detach().cpu().numpy()

        yCache.extend(y)
        probsCache.extend(probs)

        pos = np.where(y == 1)
        print(f"number of 1s in sub test {len(pos)}")
        print(f"preds that should be 1: {preds[pos]}")

        posCnt += np.sum(y)
        n_correct += np.sum(preds == y)
        n_samples += y.shape[0]
        
    acc = 100.0 * n_correct / n_samples
    print(f'Accuracy of the network on the {n_samples} validation images: {acc} %')
    print(f'Number of 1s: {posCnt}')
    writer.add_scalar('Test Accuracy', acc, 0)

    #writer.add_pr_curve('test_roc_curve', labels, probs, 0)
    writer.add_figure('Test: ROC AUC', roc_auc_graph(yCache, probsCache), 0)
    writer.close()
