import torch
from torch.functional import Tensor
from torch.optim import optimizer
from torch.utils.data import Dataset, DataLoader
import numpy as np
import torch
import torch.nn as nn
import math
from sklearn.model_selection import train_test_split
import wandb
from tqdm import tqdm

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

config = dict(data_dir = "datasets/dfExpertFeatRandomVer1_Model.npy", 
            epochs = 10, batch_size = 100, learning_rate = 1e-4, 
            dropout_prob = 0.25, test_size = .2, train_shuff = True, test_shuff = False)


def model_pipeline(hyperparameters):
    """ Pipeline flow """
    with wandb.init(project="discriminator_testing", config=hyperparameters):
      
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
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr = config.learning_rate)
    
    return model, train_loader, test_loader, criterion, optimizer

def get_data(data_dir, test_size):
    """ Load in dataset """
    df = np.load(data_dir)

    X = df[:, :-1].astype(np.float32)

    TargClasses = df[:, -1]
    y = TargClasses.astype(np.int_)

    return train_test_split(X, y, test_size = test_size)

def make_loader(X, y, batch_size, shuffle):
    dataset = DiscriminatorDataset(X, y)

    loader = DataLoader(dataset = dataset,
                            batch_size = batch_size,
                            shuffle = shuffle,
                            num_workers = 2, pin_memory= True)

    return loader

class DiscriminatorDataset(Dataset):
    """ Define dataset operations """
    def __init__(self, X, y):
        self.n_samples = X.shape[0]
        
        x1 = X[:, :3].astype(np.float32)
        x2 = X[:, 3:].astype(np.float32)

        self.x1_data = torch.from_numpy(x1)
        self.x2_data = torch.from_numpy(x2)
        self.y_data = torch.from_numpy(y)

    def __getitem__(self, index):
        sample = [self.x1_data[index], self.x2_data[index]], self.y_data[index]
        return sample

    def __len__(self):
        return self.n_samples

class ComplementLayer(nn.Module):
    """ Custom layer for ensuring complementary probabilities """
    def __init__(self, in_features, out_features):
        super(ComplementLayer, self).__init__()
        
        self.in_features = in_features
        self.out_features = out_features

        assert(self.in_features % 2 == 0)
        assert(self.out_features % 2 == 0)
        
        self.weights = nn.Parameter(torch.empty(out_features, in_features // 2).to(device))
        
        self.bias = nn.Parameter(torch.empty(1, out_features // 2).to(device))
        
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weights, a=math.sqrt(5))            
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weights)
        bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
        nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x):
        weights_flip = torch.flip(self.weights, [0,1])
        weights_cat = torch.cat((self.weights, weights_flip), 1)

        bias_flip = torch.flip(self.bias, [0])
        bias_cat = torch.cat((self.bias, bias_flip), 1)
        
        w_times_x = torch.mm(x, weights_cat.t())
        
        logit = torch.add(w_times_x, bias_cat)

        return logit

class Discriminator(nn.Module):
    """ Model definition """
    def __init__(self, dropout_prob):
        super().__init__()
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout_prob)
        self.softmax = nn.Softmax()
        
        # for feature extraction model
        self.l1 = nn.Linear(3, 20, bias=True)
        self.l2 = nn.Linear(20, 4, bias=True)

        # final layers
        self.lf1 = nn.Linear(8, 20, bias = True) #ComplementLayer(8,10)
        self.lf2 = nn.Linear(20, 2, bias = True)
    
    def forward(self, tensors):
        x1, x2 = tensors
        
        out1 = self.l1(x1)
        out1 = self.relu(out1)
        out1 = self.dropout(out1)
        out1 = self.l2(out1)
        out1 = self.relu(out1)
        out1 = self.dropout(out1)
        
        out2 = self.l1(x2)
        out2 = self.relu(out2)
        out2 = self.dropout(out2)
        out2 = self.l2(out2)
        out2 = self.relu(out2)
        out2 = self.dropout(out2)

        out2_flip = torch.flip(out2, [0])
        feats = torch.cat((out1, out2_flip), 1)
        
        out_final = self.lf1(feats)
        out_final = self.relu(out_final)
        out_final = self.dropout(out_final)
        out_final = self.lf2(out_final)
        return out_final


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
    x1, x2 = inputs
    x1, x2, labels = x1.to(device), x2.to(device), labels.to(device)
    
    outputs = model([x1, x2])
    loss = criterion(outputs, labels)
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss

def train_log(loss, example_ct, epoch):
    """ Logging for training """

    loss = float(loss)
    wandb.log({"epoch": epoch, "loss": loss}, step=example_ct)
    print(f"Loss after " + str(example_ct).zfill(5) + f" examples: {loss:.3f}")

def test(model, test_loader):
    model.eval()

    # Run the model on some test examples
    with torch.no_grad():
        cnt = 0
        correct, total = 0, 0
        for inputs, labels in test_loader:
            x1, x2 = inputs
            x1, x2, labels = x1.to(device), x2.to(device), labels.to(device)
            outputs = model([x1, x2])
            if cnt == -1:
                print(torch.nn.Softmax(outputs))
                outputs2 = model([x2,x1])
                print(torch.nn.Softmax(outputs2))
            #cnt += 1
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        print(f"Accuracy of the model on the {total} " +
              f"test images: {100 * correct / total}%")
        
        wandb.log({"test_accuracy": correct / total})

    # Save the model in the exchangeable ONNX format
    input_names = [ "input1", "input2" ]
    output_names = [ "output" ]
    torch.onnx.export(model, [x1, x2], "model.onnx", opset_version=10, do_constant_folding= False,
                        input_names=input_names, output_names=output_names)
    wandb.save("model.onnx")



wandb.login()

# run model
model = model_pipeline(config)