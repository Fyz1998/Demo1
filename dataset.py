#!/root/home/disk2/jiangwenli/anaconda3/bin/python3
# -*- coding: utf-8 -*-
# @Time : 2022-06-06
# @Author : jiangwenli
# @File : dataset.py
from torch.utils.data import Dataset
import numpy as np
import torch as t

class TimeSeriesDataset(Dataset):

    def __init__(self, params, step, scaler):


        import pandas as pd
        from scipy.signal import savgol_filter


        self.history_step = params['history_step']
        self.forecast_step = params['forecast_step']
        self.dSample = params['dSample']
        self.train_pro = params['train_pro']
        self.step = step
        self.scaler = scaler


        dataSeries = pd.read_csv('electricity1.csv')
        dataSeries = dataSeries[:]['1']

        Len = len(dataSeries)
        train_len = int(Len * self.train_pro)

        if self.step==1:
            dataSeries = dataSeries[0:train_len]
        else:
            dataSeries = dataSeries[train_len:]

        series = np.array(dataSeries[::self.dSample])
        series = savgol_filter(series, 39, 3)

        series = series.reshape(-1,1)
        scaled_values = self.scaler.fit_transform(series).astype('float32')
        self.data = scaled_values.reshape(-1, 1)

        self.len = len(self.data) - self.history_step - self.forecast_step - 1



    def __getitem__(self,index):

        index *= self.step
        start = index
        end = start + self.history_step
        X = self.data[start:end]
        y = self.data[end:end + self.forecast_step]
        return X, y



    def __len__(self):
        return self.len//self.step + 1
