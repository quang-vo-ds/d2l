#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import mylib
import time
import torch
from torch import nn
from torch.nn import functional as F
import pandas as pd
from linear_regression import LinearRegression


class KaggleHouse(mylib.DataModule):
    def __init__(self, batch_size, train=None, val=None):
        super().__init__()
        self.save_hyperparameters()
        if self.train is None:
            self.raw_train = pd.read_csv(mylib.download(
                os.path.join(mylib.DATA_URL, 'kaggle_house_pred_train.csv'), os.path.join(self.root, "KaggleHouse"),
                sha1_hash='585e9cc93e70b39160e7921475f9bcd7d31219ce'))
            self.raw_val = pd.read_csv(mylib.download(
                os.path.join(mylib.DATA_URL,'kaggle_house_pred_test.csv'), os.path.join(self.root, "KaggleHouse"),
                sha1_hash='fa19780a7b011d9b009e8bff8e99922a8ee2eb90'))
    
    def preprocess(self):
        # Remove the ID and label columns
        label = 'SalePrice'
        features = pd.concat(
            (self.raw_train.drop(columns=['Id', label]),
            self.raw_val.drop(columns=['Id'])))
        # Standardize numerical columns
        numeric_features = features.dtypes[features.dtypes!='object'].index
        features[numeric_features] = features[numeric_features].apply(
            lambda x: (x - x.mean()) / (x.std()))
        # Replace NAN numerical features by 0
        features[numeric_features] = features[numeric_features].fillna(0)
        # Replace discrete features by one-hot encoding
        features = pd.get_dummies(features, dummy_na=True)
        # Save preprocessed features
        self.train = features[:self.raw_train.shape[0]].copy()
        self.train[label] = self.raw_train[label]
        self.val = features[self.raw_train.shape[0]:].copy()
        
    def get_dataloader(self, train):
        label = 'SalePrice'
        data = self.train if train else self.val
        if label not in data: return
        get_tensor = lambda x: torch.tensor(x.values, dtype=torch.float32)
        # Logarithm of prices
        tensors = (get_tensor(data.drop(columns=[label])),  # X
                   torch.log(get_tensor(data[label])).reshape((-1, 1)))  # Y
        return self.get_tensorloader(tensors, train)
    
def k_fold_data(data, k):
    rets = []
    fold_size = data.train.shape[0] // k
    for j in range(k):
        idx = range(j * fold_size, (j+1) * fold_size)
        rets.append(KaggleHouse(data.batch_size, data.train.drop(index=idx),
                                data.train.loc[idx]))
    return rets

def k_fold(trainer, data, k, lr):
    val_loss, models = [], []
    for i, data_fold in enumerate(k_fold_data(data, k)):
        model = LinearRegression(lambd=0, lr=lr)
        model.board.yscale='log'
        if i != 0: model.board.display = False
        trainer.fit(model, data_fold)
        val_loss.append(float(model.board.data['val_loss'][-1].y))
        models.append(model)
    print(f'average validation log mse = {sum(val_loss)/len(val_loss)}')
    return models

    
if __name__ == '__main__':
    data = KaggleHouse(batch_size=64)
    data.preprocess()
    # print(data.train.shape)
    # print(data.val.shape)
    # print(data.train)
    # print(data.train.info())
    # print(data.val)
    # print(data.val.info())
    trainer = mylib.Trainer(max_epochs=10)
    models = k_fold(trainer, data, k=5, lr=0.01)
    
    # preds = [model(torch.tensor(data.val.values, dtype=torch.float32))
    #          for model in models]
    # # Taking exponentiation of predictions in the logarithm scale
    # ensemble_preds = torch.exp(torch.cat(preds, 1)).mean(1)
    # submission = pd.DataFrame({'Id':data.raw_val.Id,
    #                            'SalePrice':ensemble_preds.detach().numpy()})
    # submission.to_csv('submission.csv', index=False)

    
    # tic = time.time()
    # for X, y in data.train_dataloader():
    #     continue
    # print(f'{time.time() - tic:.2f} sec')
    
    
    # batch = next(iter(data.val_dataloader()))
    # data.visualize(batch)
    # hparams = {'num_outputs':10, 'num_hiddens_1':256, 'num_hiddens_2':256,
    #        'dropout_1':0.5, 'dropout_2':0.5, 'lr':0.1}
    # model = MLP(**hparams)
    # trainer = mylib.Trainer(max_epochs=10)
    # trainer.fit(model, data)
    
    # X, y = next(iter(data.val_dataloader()))
    # preds = model(X).argmax(axis=1)
    
    # wrong = preds.type(y.dtype) != y
    # X, y, preds = X[wrong], y[wrong], preds[wrong]
    # labels = [a+'\n'+b for a, b in zip(
    #     data.text_labels(y), data.text_labels(preds))]
    # data.visualize([X, y], labels=labels)