from pytorch_lightning.core.lightning import LightningModule
import torchmetrics
from torch.nn.functional import cross_entropy
import warnings

import torch
from torch import nn
import torch.nn.functional as F

from torch.optim import Adam
from torch.optim.lr_scheduler import CyclicLR, CosineAnnealingLR, ReduceLROnPlateau

import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

class LogisticRegression(LightningModule):
    def __init__(self, input_dim, output_dim, learning_rate = 0.0001, device='cuda', lr_scheduler='none'):
        #super(LogisticRegression, self).__init__()
        super().__init__()
        self.linear = torch.nn.Linear(input_dim, output_dim)
        self.criterion = nn.CrossEntropyLoss()
        self.learning_rate = learning_rate
        self.lr_scheduler = lr_scheduler

        self.loss_sublist = np.array([])
        self.acc_sublist = np.array([])
        #self.device = device

        # self.model = nn.Sequential(self.linear,
        #                             nn.Sigmoid())

        self.model = nn.Sequential(self.linear)

    def forward(self, x):
        outputs = self.model(x)
        return outputs

    def training_step(self, batch, batch_idx):
        embedding, labels, filename = batch
        #embedding, labels = embedding.to(self.device).squeeze(1), labels.to(self.device).squeeze(1)
        embedding, labels = embedding.squeeze(1), labels    #[batch_size, input_dim], [batch_size, 1]
        labels = labels.to(torch.int64)

        #Forward pass through the model
        #outputs = torch.sigmoid(self.linear(embedding))
        outputs = self.model(embedding)         # [Batch_Size, Num_classes]
        
        #Calculate loss
        loss = self.criterion(outputs, labels.squeeze(1))

        #Calculate and log training metrics

        preds = torch.exp(outputs.cpu().data)/torch.sum(torch.exp(outputs.cpu().data))
        #preds = torch.argmax(outputs, dim=1)
        #print("Labels: ", labels)
        #print("Predictions: ", preds)
        #print("Unique Predictions: ", np.unique(preds.cpu().numpy()))
        #acc = torchmetrics.functional.accuracy(preds, torch.squeeze(labels, dim=1))
        acc = np.array(np.argmax(preds, axis=1) == labels.cpu().data.view(-1)).astype('int')
        #print("Accuracy: ", acc)

        self.loss_sublist = np.append(self.loss_sublist, loss.cpu().detach().numpy())
        self.acc_sublist = np.append(self.acc_sublist, acc, axis=0)
        
        #Logging metrics
        self.log('train_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log('train_acc', torch.from_numpy(acc), on_step=False, on_epoch=True, prog_bar=True)

        return loss

    def training_epoch_end(self, training_step_outputs):
        #print("Train Loss: ", training_step_outputs['train_loss'])
        #print("Train Acc: ", training_step_outputs['train_acc'])
        
        #print(training_step_outputs)
        print("Train acc: {} | Train Loss: {}".format(np.mean(self.acc_sublist), np.mean(self.loss_sublist)))
        print("\n")

    def test_step(self, batch, batch_idx):
        test_embd, test_labels, filename = batch
        #test_embd, test_labels = test_embd.to(self.device).squeeze(1), test_labels.to(self.device)
        test_embd, test_labels = test_embd.squeeze(1), test_labels
        test_labels = test_labels.to(torch.int64)

        test_output = self.model(test_embd) 
        _, predicted = torch.max(test_output, dim=1)

        test_loss = self.criterion(test_output, test_labels.squeeze(1))

        #Calculate and log training metrics
        test_preds = torch.argmax(test_output, dim=1)
        test_acc = torchmetrics.functional.accuracy(test_preds, torch.squeeze(test_labels, dim=1))

        #Logging metrics
        self.log('test_loss', test_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log('test_acc', test_acc, on_step=False, on_epoch=True, prog_bar=True)

        return test_loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)

        if(self.lr_scheduler == 'none'):
            return optimizer
        elif(self.lr_scheduler == 'reduce_plateau'):
            scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, min_lr=1e-6, patience=10, verbose=True)

        return {"optimizer": optimizer, 
                "lr_scheduler":{
                    "scheduler": scheduler,
                    "monitor": 'train_acc'
                }}



        

