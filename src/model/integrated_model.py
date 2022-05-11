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

class IntegratedModel(LightningModule):
    def __init__(self, input_dim, output_dim, batch_size, encoder, learning_rate = 0.0001, device='cuda', lr_scheduler='none'):

        super().__init__()
        self.linear = torch.nn.Linear(input_dim, output_dim)
        self.encoder = encoder

        self.criterion = nn.CrossEntropyLoss()
        self.learning_rate = learning_rate
        self.lr_scheduler = lr_scheduler
        self.batch_size = batch_size

        self.loss_sublist = np.array([])
        self.acc_sublist = np.array([])
        
        self.encoder.eval()
        self.model = nn.Sequential(self.linear)

        for p in self.encoder.parameters():
            p.requires_grad = False

    def training_step(self, batch, batch_idx):
        image, labels, filename = batch

        #Pass through the encoder to create the embedding
        embedding = self.encoder(image)                #[batch_size, input_dim]
        labels = torch.unsqueeze(labels, 1)     #[batch_size, 1]

        #Forward pass through the Linear Layer
        outputs = self.model(embedding)         # [Batch_Size, Num_classes]

        #Calculate Loss
        loss = self.criterion(outputs, labels.squeeze(1))

        #Predictions
        preds = torch.softmax(outputs, dim=1)

        acc = np.array(np.argmax(preds.cpu().detach().numpy(), axis=1) == labels.cpu().data.view(-1).numpy()).astype('int')

        self.loss_sublist = np.append(self.loss_sublist, loss.cpu().detach().numpy())
        self.acc_sublist = np.append(self.acc_sublist, acc, axis=0)
        
        #Logging metrics
        self.log('train_loss', loss, on_step=False, on_epoch=True, prog_bar=True, batch_size=self.batch_size)
        self.log('train_acc', torch.from_numpy(acc), on_step=False, on_epoch=True, prog_bar=True, batch_size=self.batch_size)

        return loss

    def training_epoch_end(self, training_step_outputs):
        
        print("Train acc: {} | Train Loss: {}".format(np.mean(self.acc_sublist), np.mean(self.loss_sublist)))
        
        print("\n")

    def test_step(self, batch, batch_idx):
        image, labels, filename = batch

        #Pass through the encoder to create the embedding
        embedding = self.encoder(image)                #[batch_size, input_dim]
        labels = torch.unsqueeze(labels, 1)            #[batch_size, 1]

        #Forward pass through the Linear Layer
        outputs = self.model(embedding)         # [Batch_Size, Num_classes]

        #Calculate Loss
        loss = self.criterion(outputs, labels.squeeze(1))

        preds = torch.softmax(outputs, dim=1)

        acc = np.array(np.argmax(preds.cpu().detach().numpy(), axis=1) == labels.cpu().data.view(-1).numpy()).astype('int')

        #Logging metrics
        self.log('test_loss', loss, on_step=False, on_epoch=True, prog_bar=True, batch_size=self.batch_size)
        self.log('test_acc', torch.from_numpy(acc), on_step=False, on_epoch=True, prog_bar=True, batch_size=self.batch_size)

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








