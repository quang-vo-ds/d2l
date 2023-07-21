#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun 18 10:58:58 2023

@author: quangvo
"""

import mylib
import torch
from torch import nn
import random

import os

os.environ['KMP_DUPLICATE_LIB_OK']='True'


class Data(mylib.DataModule):
    """Synthetic data for linear regression."""
    def __init__(self, num_train=20, num_val=100, num_inputs=200, batch_size=5):
        super().__init__()
        self.save_hyperparameters()
        n = num_train + num_val
        self.X = torch.randn(n, num_inputs)
        noise = torch.randn(n, 1) * 0.01
        w, b = torch.ones((num_inputs, 1)) * 0.01, 0.05
        self.y = torch.matmul(self.X, w) + b + noise
    
    def get_dataloader(self, train):
        i = slice(0, self.num_train) if train else slice(self.num_train, None)
        return self.get_tensorloader((self.X, self.y), train, i)
    
def l2_penalty(w):
    return (w**2).sum()/2

def train_scratch(lambd):
    model = LinearRegression(lambd=lambd, lr=0.01)
    model.board.yscale='log'
    trainer.fit(model, data)
    w, b = model.get_w_b()
    print('L2 norm of w:', float(l2_penalty(w)))
    

class LinearRegression(mylib.Module):
    """The linear regression model implemented with high-level APIs."""
    def __init__(self, lambd, lr):
        super().__init__()
        self.save_hyperparameters()
        self.net = nn.LazyLinear(1)
        self.net.weight.data.normal_(0, 0.01)
        self.net.bias.data.fill_(0)
    
    def forward(self, X):
        return self.net(X)
    
    def loss(self, y_hat, y):
        fn = nn.MSELoss()
        return fn(y_hat, y)
    
    def configure_optimizers(self):
        return torch.optim.SGD([
            {'params': self.net.weight, 'weight_decay': self.lambd},
            {'params': self.net.bias}], lr=self.lr)
    
    def get_w_b(self):
        return (self.net.weight.data, self.net.bias.data)
        
        
if __name__ == '__main__':
    data = Data(num_train=20, num_val=100, num_inputs=200, batch_size=5)
    trainer = mylib.Trainer(max_epochs=10)
    model = LinearRegression(lambd=3, lr=0.01)
    model.board.yscale='log'
    trainer.fit(model, data)
    
    print('L2 norm of w:', float(l2_penalty(model.get_w_b()[0])))