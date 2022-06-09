#!/root/home/disk2/jiangwenli/anaconda3/bin/python3
# -*- coding: utf-8 -*-
# @Time : 2022-06-06
# @Author : jiangwenli
# @File : model.py
import torch as t

class Forcast(t.nn.Module):

    def __init__(self, history_step=20,forecast_step=10,hidden_size=64,num_layers=2,dropout_rate=0.2):
        super(Forcast, self).__init__()

        self.history_step = history_step
        self.forecast_step = forecast_step
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout_rate = dropout_rate


        # self.lstm = t.nn.Sequential(
        #     t.nn.LSTM(
        #         input_size = 1,
        #         hidden_size = self.hidden_size,
        #         num_layers = self.num_layers
        #     )
        # )


        self.conv1 = t.nn.Sequential(
            t.nn.Conv2d(
                in_channels=self.history_step,
                out_channels=32,
                kernel_size=5,
                stride=1,
                padding=2,
                dilation=1
            ),
            t.nn.ReLU(),
            # t.nn.MaxPool2d(kernel_size=2),
        )


        self.layNorm = t.nn.Sequential(
            t.nn.LayerNorm(
                normalized_shape = [32,1,1],
                eps = 1e-5
            ),
            # t.nn.ReLU(),
        )



        self.conv2 = t.nn.Sequential(
            t.nn.Conv2d(
                in_channels=32,
                out_channels=64,
                kernel_size=3,
                stride=1,
                padding=1,
                dilation=1
            ),
            t.nn.ReLU(),
            # t.nn.MaxPool2d(kernel_size=2),
        )

        self.ful1 = t.nn.Linear(64, 32)
        self.drop = t.nn.Dropout(self.dropout_rate)
        self.ful2 = t.nn.Sequential(t.nn.Linear(32, self.forecast_step))


    def forward(self,x):

        # x = self.lstm(x)
        # x = t.unsqueeze(x, 3)
        x = self.conv1(x)
        x = self.layNorm(x)
        x = self.conv2(x)
        x = x.view(x.size(0), -1)
        x = self.ful1(x)
        x = self.drop(x)
        output = self.ful2(x)
        return output

