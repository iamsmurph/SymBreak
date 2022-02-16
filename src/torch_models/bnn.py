
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

import numpy as np
import pandas as pd
import pickle

from blitz.modules import BayesianLinear
from blitz.utils import variational_estimator

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

from tqdm import tqdm
import wandb
import pickle


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
round0_dir = "datasets/round_0/combined/round0_only_train.csv"
round1_dir = "datasets/round_1/combined/train_cumulative.csv"
cum_dir = "datasets/round_1/combined/round1_only_train.csv"
####### DATA HANDLING #######
config = dict(data_dir = cum_dir, 
                input_size = 10, 
                output_size = 1, 
                epochs = 500, 
                batch_size = 64, 
                learning_rate = 1e-3, 
                test_size = .1, 
                train_shuff = True, 
                test_shuff = False)

run = wandb.init(project="bayesian_NN", config=config)

def get_data(data_dir, test_size):
    """ Load in dataset """
    labeled_df = pd.read_csv(data_dir)
    X, y = labeled_df.iloc[:, :-1], labeled_df.iloc[:, -1]
    scale = StandardScaler()
    X = scale.fit_transform(X)
    #pickle.dump(scale, open("modeling/torch_models/bnn_scaler.pkl", "wb"))
    y = np.expand_dims(y, -1)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)
    X_train, y_train = torch.tensor(X_train).float(), torch.tensor(y_train).float()
    X_test, y_test = torch.tensor(X_test).float(), torch.tensor(y_test).float()
    
    return X_train, X_test, y_train, y_test

def make_loader(X, y, batch_size, shuffle):
    dataset = TensorDataset(X, y)

    loader = DataLoader(dataset = dataset,
                            batch_size = batch_size,
                            shuffle = shuffle,
                            num_workers = 2, 
                            pin_memory= True)

    return loader

####### MODELING #######
@variational_estimator
class BayesianRegressor(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.blinear1 = BayesianLinear(input_dim, 100)
        self.blinear2 = BayesianLinear(100, output_dim)
        
    def forward(self, x):
        x_ = self.blinear1(x)
        x_ = F.relu(x_)
        out = self.blinear2(x_)
        return out

def train_log(loss, example_ct, epoch):
    loss = float(loss)
    wandb.log({"epoch": epoch, "training loss": loss}, step=example_ct)
    print(f"Loss after " + str(example_ct).zfill(5) + f" examples: {loss:.3f}")

def train_batch(inputs, labels, model, optimizer, criterion):
    inputs, labels = inputs.to(device), labels.to(device)
    
    optimizer.zero_grad()
    
    loss = model.sample_elbo(inputs=inputs,
                        labels=labels,
                        criterion=criterion,
                        sample_nbr=3)
    loss.backward()
    optimizer.step()

    return loss

def train(model, loader, criterion, optimizer, config):
    model.train()
    wandb.watch(model, criterion, log="all", log_freq=10)

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

            if ((batch_ct + 1 ) % 50) == 0:
                ic_acc, under_ci_upper, over_ci_lower = evaluate(model,
                                                                inputs,
                                                                labels,
                                                                samples=25, 
                                                                std_multiplier = 1.96)
            
                wandb.log({"ic_acc": ic_acc, "inside_upper": under_ci_upper, "inside_lower": over_ci_lower})

####### TESTING #######
def evaluate(regressor,
            X,
            y,
            samples = 100,
            std_multiplier = 1.96):
    preds = [regressor(X) for i in range(samples)]
    preds = torch.stack(preds)
    means = preds.mean(axis=0)
    stds = preds.std(axis=0)
    ci_upper = means + (std_multiplier * stds)
    ci_lower = means - (std_multiplier * stds)
    ic_acc = (ci_lower <= y) * (ci_upper >= y)
    ic_acc = ic_acc.float().mean()
    return ic_acc, (ci_upper >= y).float().mean(), (ci_lower <= y).float().mean()

def sample_prediction(regressor, X, samples = 100):
    preds = [regressor(X) for _ in range(samples)]
    preds = torch.stack(preds)
    means = preds.mean(axis=0)
    return means

def test(model, loader):
    model.eval()

    # Run the model on some test examples
    with torch.no_grad():
        cnt = 0
        mse_scores = 0
        for datapoints, labels in loader:
            means = sample_prediction(model, datapoints) #torch.Tensor(Xtest_scaled).float()
            mse_score = mean_squared_error(means.detach().numpy(), labels.detach().numpy())
            mse_scores += mse_score
            wandb.log({"test batch loss": mse_score})
            
####### RUN #########
def make(config):
    """ Pretraining preparation """
    # Make the data
    X_train, X_test, y_train, y_test = get_data(config.data_dir, config.test_size)
    train_loader = make_loader(X_train, y_train, config.batch_size, config.train_shuff)
    test_loader = make_loader(X_test, y_test, config.batch_size, config.test_shuff)

    # Make the model
    model = BayesianRegressor(config.input_size, config.output_size)

    # Make loss and optimizer
    criterion = torch.nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)
    
    return model, train_loader, test_loader, criterion, optimizer

def model_pipeline(hyperparameters):
    """ Pipeline flow """
    with wandb.init(project="Bayesian_NN", config=hyperparameters):
      
      config = wandb.config

      model, train_loader, test_loader, criterion, optimizer = make(config)

      train(model, train_loader, criterion, optimizer, config)

      test(model, test_loader)

    return model

if __name__=="__main__":
    model = model_pipeline(config)
    #torch.save(model.state_dict(), "modeling/torch_models/bnn")
