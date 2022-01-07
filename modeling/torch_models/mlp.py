import os
import pandas as pd
import pathlib
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from sklearn.model_selection import train_test_split
import sys
from sklearn.preprocessing import StandardScaler
#device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')

# Data Handling
class Dataset(Dataset):
    def __init__(self, df):
       self.df = df
        
    def __len__(self):
        return self.df.shape[0]
    
    def __getitem__(self, idx):
        tuple = self.df[idx]
        feats = tuple[:-1]
        y = tuple[-1]
        feats, y = torch.tensor(feats).float(), torch.tensor(y).float()
        return feats, y

class DataModule(pl.LightningDataModule):
    def __init__(self, data_dir, batch_size): #batch_size
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
    
    def setup(self, stage):
        df = pd.read_csv(os.path.join(self.data_dir, "combined_pool_dens_grad_dipole_stdScl_rot.csv"))
        train_df, test_df = train_test_split(df.values, test_size=0.1, random_state=42)
        train_df, val_df = train_test_split(train_df, test_size = 0.2, random_state=42)
        self.train_dataset = Dataset(train_df)
        self.val_dataset = Dataset(val_df)
        self.test_dataset = Dataset(test_df)

    def train_dataloader(self):
        train_dl = DataLoader(self.train_dataset, 
                              batch_size=self.batch_size, 
                              shuffle=True, 
                              num_workers= 4)
        return train_dl
    def val_dataloader(self):
        val_dl = DataLoader(self.val_dataset, 
                            batch_size= self.batch_size,
                            num_workers= 4)
        return val_dl
    def test_dataloader(self):
        test_dl = DataLoader(self.test_dataset, 
                             batch_size=self.batch_size, 
                             num_workers= 4)
        return test_dl

# Models  
class MLP(pl.LightningModule):
    def __init__(self, n_in, n_out, lr, drop_p):
        super().__init__()

        self.dropout = nn.Dropout(drop_p)
        self.relu = nn.ReLU()
        self.lr = lr

        self.fc1 = nn.Linear(n_in, n_in, bias = True)
        self.fc2 = nn.Linear(n_in, n_out, bias = True)
        
    def forward(self, x):
        print(x.shape)
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr = self.lr)
        return optimizer

    def training_step(self, train_batch, batch_idx):
        x, y = train_batch
        preds = self.forward(x)
        mse = nn.MSELoss()
        loss = mse(preds, torch.unsqueeze(y, 1))
        self.log('train_loss', loss)
        return loss
    
    def validation_step(self, val_batch, batch_idx):
        x, y = val_batch
        preds = self.forward(x)
        mse = nn.MSELoss()
        loss = mse(preds, torch.unsqueeze(y, 1))
        self.log('val_loss', loss)

    def test_step(self, test_batch, batch_idx):
        x, y = test_batch
        preds = self.forward(x)
        mse = nn.MSELoss()
        loss = mse(preds, torch.unsqueeze(y,1))
        self.log("test_loss", loss)

### MAIN ###
if __name__ == "__main__":
    data_root = pathlib.Path('datasets/round_1/combined')
    #n_epochs = 100

    # Data loaders
    batch_size = 64

    # Model building
    n_in = 50
    n_out = 1
    lr = 1e-3
    drop_p = 0.2
    n_gpus = 1
    max_epochs = 70

    
    wandb_logger = WandbLogger(project="MeanPoolingOrganoids")
    data = DataModule(data_root, batch_size)
    model = MLP(n_in, n_out, lr, drop_p)
    trainer = pl.Trainer(gpus=n_gpus, max_epochs=max_epochs, logger=wandb_logger, callbacks=[EarlyStopping(monitor="val_loss")])
    trainer.fit(model, data)
    trainer.test(model, data)