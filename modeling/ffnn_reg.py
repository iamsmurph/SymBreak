import torch
from torch.functional import Tensor
from torch.optim import optimizer
from torch.utils.data import Dataset, DataLoader
import numpy as np
import torch
import torch.nn as nn
import math
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
import wandb
from tqdm import tqdm

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

config = dict(data_dir = "datasets/big_model.npy", 
            epochs = 1000, batch_size = 50, learning_rate = 1e-5, 
            dropout_prob = 0.25, test_size = .2, train_shuff = True, 
            test_shuff = False, weight_decay = 0.9)


def model_pipeline(hyperparameters):
    """ Pipeline flow """
    with wandb.init(project="discriminator_reg_testing", config=hyperparameters):
      
      config = wandb.config

      model, train_loader, test_loader, criterion, optimizer = make(config)

      train(model, train_loader, criterion, optimizer, config)

      test(model, test_loader)

    return model

def make(config):
    """ Pretraining preparation """
    # Make the data
    X_train, X_test, y_train, y_test = get_data(config.data_dir, config.test_size)

    train_loader = make_loader(X_train, y_train, config.batch_size, config.train_shuff)
    test_loader = make_loader(X_test, y_test, config.batch_size, config.test_shuff)

    # Make the model
    model = Discriminator(config.dropout_prob).to(device)

    # Make the loss and optimizer
    criterion = torch.nn.L1Loss()
    optimizer = torch.optim.Adam(model.parameters(), lr = config.learning_rate)
    
    return model, train_loader, test_loader, criterion, optimizer

def get_data(data_dir, test_size):
    """ Load in dataset """
    df = np.load(data_dir).astype(np.float32)

    X = df[:, :-1]

    TargClasses = df[:, -1]
    y = TargClasses

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = test_size)

    scaler = preprocessing.StandardScaler().fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)

    return X_train, X_test, y_train, y_test

def make_loader(X, y, batch_size, shuffle):
    dataset = DiscriminatorDataset(X, y)

    loader = DataLoader(dataset = dataset,
                            batch_size = batch_size,
                            shuffle = shuffle,
                            num_workers = 2)

    return loader

class DiscriminatorDataset(Dataset):
    """ Define dataset operations """
    def __init__(self, X, y):
        self.n_samples = X.shape[0]

        self.x_data = torch.from_numpy(X)
        self.y_data = torch.from_numpy(y)

    def __getitem__(self, index):
        sample = self.x_data[index], self.y_data[index]
        return sample

    def __len__(self):
        return self.n_samples

class Discriminator(nn.Module):
    """ Model definition """
    def __init__(self, dropout_prob):
        super().__init__()
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout_prob)
        
        # for feature extraction model
        self.l1 = nn.Linear(24, 1, bias=True)
        self.l2 = nn.Linear(10, 1, bias=True)
        self.lf = nn.Linear(5, 1, bias=True)
    
    def forward(self, x):
        out = self.l1(x)
        out = self.relu(out)
        return out

def train(model, loader, criterion, optimizer, config):
    """ Training and logging """
    model.train()
    wandb.watch(model, criterion, log="all", log_freq=10)

    #total_batches = len(loader) * config.epochs
    example_ct = 0  # number of examples seen
    batch_ct = 0
    for epoch in tqdm(range(config.epochs)):
        for _, (inputs, labels) in enumerate(loader):

            loss = train_batch(inputs, labels, model, optimizer, criterion)
            example_ct +=  len(inputs)
            batch_ct += 1

            # Report metrics every 25th batch
            if ((batch_ct + 1) % 25) == 0:
                train_log(loss, example_ct, epoch)


def train_batch(inputs, labels, model, optimizer, criterion):
    """ Batch updating """
    inputs, labels = inputs.to(device), labels.to(device)
    
    outputs = model(inputs)
    loss = criterion(outputs, labels)
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss

def train_log(loss, example_ct, epoch):
    """ Logging for training """

    loss = float(loss)
    wandb.log({"Num Epoch": epoch, "Training Loss": loss}, step=example_ct)
    print(f"Loss after " + str(example_ct).zfill(5) + f" examples: {loss:.3f}")

def test(model, test_loader):
    model.eval()

    # Run the model on some test examples
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = nn.L1Loss()
            eval = loss(inputs, outputs)
        
        wandb.log({"Test L1 Loss": eval})

    # Save the model in the exchangeable ONNX format
    torch.onnx.export(model, inputs, "model.onnx") #, opset_version=10, do_constant_folding= False
    wandb.save("model.onnx")



wandb.login()

# run model
model = model_pipeline(config)