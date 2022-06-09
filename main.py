#!/root/home/disk2/jiangwenli/anaconda3/bin/python3
# -*- coding: utf-8 -*-
# @Time : 2022-06-06
# @Author : jiangwenli
# @File : main.py

import time
import matplotlib.pyplot as plt
import numpy as np
import torch as t
from sklearn.metrics import mean_absolute_error, mean_squared_error
from torch.utils.data import DataLoader
from tqdm import *
from dataset import TimeSeriesDataset
from model import Forcast
from SCINet import SCINet
from sklearn.preprocessing import MinMaxScaler


time_start = time.time()

def Metrics(pred, label):
    pred = np.array(pred).reshape(-1, 1)
    label = np.array(label).reshape(-1,1)
    mae = mean_absolute_error(label, pred)
    mse = mean_squared_error(label, pred)
    rmse = np.sqrt(mse)
    t1 = np.sum((pred - label) ** 2) / np.size(label)
    t2 = np.sum(abs(label)) / np.size(label)
    nrmse = np.sqrt(t1) / t2
    return np.array([mae, mse, rmse, nrmse])


params = {
    'history_step': 16,
    'forecast_step': 8,
    'hidden_size': 64,
    'num_layers': 2,
    'dropout_rate': 0.5,
    'epochs': 10,
    'batch_size': 16,
    'lr': 0.001,
    'train_pro':0.8,
    'dSample':10
}


# model = Forcast(params['history_step'],
#                params['forecast_step'],
#                params['hidden_size'],
#                params['num_layers'],
#                params['dropout_rate']
# )

model = SCINet(
    output_len=8,
    input_len=16,
    input_dim=1,
    hid_size=1,
    num_stacks=1,
    num_levels=3,
    num_decoder_layer=1,
    concat_len=0,
    groups=1,
    kernel=5,
    dropout=0.5,
    single_step_output_One=0,
    positionalE=False,
    modified=True,
    RIN=False
)

# model = model.type(t.FloatTensor)
# from torchsummary import summary
# summary(model, input_size=(16,1,1), batch_size=16, device="cpu")


optim = t.optim.AdamW(model.parameters(), lr=params['lr'],weight_decay=0.01)
schedule = t.optim.lr_scheduler.CosineAnnealingLR(optim, 5)


scaler = MinMaxScaler(feature_range=(-1, 1))
trainset = TimeSeriesDataset(params, 1, scaler)
testset = TimeSeriesDataset(params, params['forecast_step'], scaler)

trainLoader = DataLoader(trainset, params['batch_size'], shuffle= False, drop_last=True)
testLoader = DataLoader(testset, params['batch_size'], shuffle= False, drop_last=True)



lossfunc = t.nn.MSELoss()

def train():
    Metric = []
    for epoch in range(params['epochs']):
        model.train()
        for X, y in tqdm(trainLoader):
            optim.zero_grad()
            # X = t.unsqueeze(X, 3)
            prediction = model(X)
            prediction = prediction.reshape(y.shape)
            loss = lossfunc(prediction, y)
            loss.backward()
            optim.step()

        metrics = np.zeros((4,))

        model.eval()
        i = 0
        forecasts = list()
        oridata = list()
        for X, y in testLoader:
            i+=1
            # X = t.unsqueeze(X, 3)
            prediction = model(X)
            prediction = prediction.reshape(y.shape)
            prediction = prediction.detach()
            y = y.detach()

            prediction = scaler.inverse_transform(prediction.squeeze())
            y = scaler.inverse_transform(y.squeeze())

            if epoch == params['epochs'] - 1:
                forecasts.append(np.array(prediction))
                oridata.append(np.array(y))

            metrics += Metrics(prediction, y)

        print(f'Epoch-{epoch}: MAE={metrics[0] / len(testLoader)};MSE={metrics[1] / len(testLoader)};'
              f'RMSE={metrics[2] / len(testLoader)};NRMSE={metrics[3] / len(testLoader)}')

        Metric.append(metrics)

        if epoch == params['epochs']-1:

            forecasts = np.array(forecasts).reshape(-1,1)
            oridata = np.array(oridata).reshape(-1, 1)
            print("forecasts's shape is:",forecasts.shape)
            print("oridata's shape is:", oridata.shape)
            plt.figure(1, figsize=(12, 4))
            plt.plot(oridata[:200], label="Test_data")
            plt.plot(forecasts[:200],label="Foreacast_data")
            plt.title(f"On Epochs-{epoch} Testing Plot")
            plt.legend()
            plt.show()
    return Metric


Metric  = train()
time_end = time.time()
print('time consumption:',time_end-time_start)

plt.figure(1, figsize=(12, 4))
plt.subplot(221)
plt.plot(np.array(Metric)[:,0],label='MAE')
plt.subplot(222)
plt.plot(np.array(Metric)[:,1],label='MSE')
plt.subplot(223)
plt.plot(np.array(Metric)[:,2],label='RMSE')
plt.subplot(224)
plt.plot(np.array(Metric)[:,3],label='NRMSE')
plt.legend()
plt.show()