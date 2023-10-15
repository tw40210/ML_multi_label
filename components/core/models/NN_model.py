import collections

import tqdm

from core.models.basic_models import BasicModel
import numpy as np
from xgboost import XGBRegressor
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, accuracy_score
import warnings
import pandas as pd
import torch
from torch import nn, optim
from typing import Dict
from tensorboardX import SummaryWriter
import core
from datetime import datetime

warnings.filterwarnings('ignore', category=FutureWarning)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = 'cpu'
print('Using device:', device)


class NNModel(BasicModel):
    def __init__(self):
        super(NNModel, self).__init__()
        self.net = None
        self.net_cluster = None
        current_time = datetime.now().strftime("%b%d-%H-%M")
        self.logger = SummaryWriter(log_dir=core.REPO_PATH+f"/runs/{current_time}")
        self.train_mean = None
        self.train_std = None
        self.best_score = {"train": {}, "valid": {}}

    def train(self, params, valid_sets, train_set, callbacks, names:Dict={"x":'data', "y":'labels'}):
        num_epoch = params["num_epoch"] if "num_epoch" in params else 10
        loss_fn  = params["loss_fn "] if "loss_fn " in params else nn.BCELoss()
        batch_size = params["batch_size "] if "batch_size " in params else 32
        lr =params["lr "] if "lr " in params else 1e-3
        metric = params["metric"]
        self.best_score["train"][metric]=np.inf
        self.best_score["valid"][metric]=np.inf



        x = train_set[names["x"]]
        y = train_set[names["y"]]
        train_dataset = NNDataset(x, y)
        dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        self.train_mean=train_dataset.mean
        self.train_std=train_dataset.std

        valid_datasets=[]
        for valid_set in valid_sets:
            valid_x = valid_set[names["x"]]
            valid_y = valid_set[names["y"]]
            dataset = NNDataset(valid_x, valid_y, mean=self.train_mean, std = self.train_std)
            valid_datasets.append(dataset)

        # self.net = Net(train_dataset.x.shape[1], train_dataset.y.shape[1]).to(device)
        self.net = NetCluster(10, train_dataset.x.shape[1], train_dataset.y.shape[1])
        optimizer = optim.Adam(self.net.parameters(), lr=lr)

        total_step = 1
        # # Train the model
        for epoch in range(num_epoch):

            tot_loss = 0
            step = 1
            with tqdm.tqdm(dataloader) as bar:
                for x, y in bar:
                  y_hat = self.net(x)
                  loss = loss_fn(y_hat, y)

                  optimizer.zero_grad()
                  loss.backward()
                  optimizer.step()
                  tot_loss+=loss.item()

                  if step%10==0:
                      avg_loss = tot_loss / step
                      bar.set_postfix(loss= loss.item(), avg_loss = avg_loss)
                      self.logger.add_scalar('loss', loss.item(), global_step=total_step)
                      self.best_score["train"][metric] = min(self.best_score["train"][metric], avg_loss)
                  step+=1
                  total_step+= 1

            for valid_dataset in valid_datasets:
                x, y = valid_dataset.get_all()
                y_hat = self.net(x)
                loss = loss_fn(y_hat, y)
                bar.write(f"validation loss: {loss}")
                self.best_score["valid"][metric] = min(self.best_score["valid"][metric], loss.item())



        return self
    def predict(self, data: pd.DataFrame) ->np.ndarray:
        x =torch.tensor(data.to_numpy())
        x = (x-self.train_mean) / self.train_std
        x = x.float().to(device)
        y_hat = self.net(x)

        if y_hat.shape[-1]==1:
            y_hat:torch.Tensor = y_hat.squeeze()

        return y_hat.detach().cpu().numpy()

class NetCluster(nn.Module):
    def __init__(self, num_net, in_ch, out_ch):
        super(NetCluster, self).__init__()
        self.net1=Net(in_ch, out_ch).to(device)
        self.net2=Net(in_ch, out_ch).to(device)

    def forward(self, x):
        y1 = self.net1(x)
        y2 = self.net2(x)
        return (y1+y2)/2




class Net(nn.Module):
  def __init__(self, input_size, output_size):
    super(Net, self).__init__()
    self.linear1 = nn.Linear(input_size, 128)
    self.linear2 = nn.Sequential(
                        nn.Linear(128, 128),
                        nn.BatchNorm1d(128),
                        nn.LeakyReLU(),
                        nn.Linear(128, 64),
                        nn.BatchNorm1d(64),
                    )
    self.linear3 = nn.Linear(64, output_size)
    self.relu = nn.LeakyReLU()
    self.sig = nn.Sigmoid()

  def forward(self, x):
    x = self.linear1(x)
    x = self.relu(x)
    x = self.linear2(x)
    x = self.relu(x)
    x = self.linear3(x)
    x = self.sig(x)
    return x

class NNDataset(torch.utils.data.Dataset):
  def __init__(self, x, y, mean=None, std=None):
    if len(y.shape) == 1:
      y = y[:, None]

    self.mean = torch.nan_to_num(x.mean(dim=0), nan=0.0) if mean is None else mean
    self.std = x.std(dim=0) if std is None else std
    self.std = torch.where(self.std==0, 1, self.std)
    self.x = (x-self.mean) /self.std
    self.x = self.x.to(device)
    self.y = y.to(device)
    self.threshold = y.float().mean()


  def __getitem__(self, idx):
    return self.x[idx].float(), self.y[idx].float()

  def __len__(self):
    return self.x.shape[0]

  def get_all(self):
      return self.x.float(), self.y.float()

# Load the dataframe
# df = pd.read_csv('data.csv')
#
# # Convert the dataframe to PyTorch tensors
# X = torch.tensor(df.to_numpy()[:, :-1])
# y = torch.tensor(df.to_numpy()[:, -1])
#
# # Define the model and optimizer
# model = Net(X.shape[1], y.shape[1])
# optimizer = optim.Adam(model.parameters())
#
# # Train the model
# for epoch in range(100):
#   loss = model(X) - y
#   loss = loss.mean()
#
#   optimizer.zero_grad()
#   loss.backward()
#   optimizer.step()
#
# # Evaluate the model
# predictions = model(X)
# accuracy = (predictions > 0.5) == (y > 0.5)
# accuracy = accuracy.mean()
#
# print('Accuracy:', accuracy)