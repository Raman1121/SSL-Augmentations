from pytorch_lightning.core.lightning import LightningModule
import torchmetrics
from torch.nn.functional import cross_entropy
import warnings

import torch
from torch import nn
import torch.nn.functional as F

from torch.optim import Adam
from torch.optim.lr_scheduler import CyclicLR, CosineAnnealingLR, ReduceLROnPlateau

from sklearn.metrics import accuracy_score, f1_score
import numpy as np
import os

class IntegratedModel(LightningModule):
    def __init__(self, input_dim, output_dim, batch_size, encoder, class_weights,
                 learning_rate = 0.0001, device='cuda', lr_scheduler='none',
                 activation='softmax', multilable=False, criterion='cross_entropy'):

        super().__init__()
        self.linear = torch.nn.Linear(input_dim, output_dim)
        self.encoder = encoder

        #self.criterion = nn.CrossEntropyLoss()
        self.learning_rate = learning_rate
        self.lr_scheduler = lr_scheduler
        self.batch_size = batch_size
        self.activation = activation
        self.multilable = multilable
        self.class_weights = class_weights

        self.loss_sublist = np.array([])
        self.acc_sublist = np.array([])
        
        self.encoder.eval()
        self.model = nn.Sequential(self.linear)

        if(criterion == 'cross_entropy'):
            self.criterion = nn.CrossEntropyLoss(weight=self.class_weights)
        elif(criterion == 'bce'):
            self.criterion = nn.BCELoss()

        for p in self.encoder.parameters():
            p.requires_grad = False

    def calculate_acc(self, probs, true_labels, multilable=False):

        acc = -9999

        if(multilable):
            #TODO: Calculate class-wise accuracy

            N,C = true_labels.shape

            probs = probs.cpu().detach().numpy()
            true_labels = true_labels.cpu().detach().numpy()

            for prob in probs:
                prob[prob >= 0.5] = 1
                prob[prob < 0.5] = 0
            
            
            #acc = sum((probs == true_labels).astype('int') * true_labels)/len(probs)
            acc = (probs == true_labels).sum()/(N*C)

        else:
            _arr, _counts = np.unique(np.argmax(probs.cpu().detach().numpy(), axis=1), return_counts=True)
            print(_arr, _counts/_counts.sum()*100)
            #print(np.unique(true_labels.cpu().data.view(-1).numpy(), return_counts=True))
            acc = np.array(np.argmax(probs.cpu().detach().numpy(), axis=1) == true_labels.cpu().data.view(-1).numpy()).astype('int').sum().item() / probs.size(0)
            
        return acc

    def calculate_f1(self, probs, true_labels, multilable=False):

        if(multilable):
            N,C = true_labels.shape

            probs = probs.cpu().detach().numpy()
            true_labels = true_labels.cpu().detach().numpy()

            for prob in probs:
                prob[prob >= 0.5] = 1
                prob[prob < 0.5] = 0

            f1 = f1_score(true_labels, probs, average='weighted')

        else:
            predicted_labels = np.argmax(probs.cpu().detach().numpy(), axis=1)
            true_labels = true_labels.cpu().data.view(-1).numpy()

            f1 = f1_score(true_labels, predicted_labels, average='weighted')

        return f1

    def training_step(self, batch, batch_idx):
        image, labels = batch

        #Pass through the encoder to create the embedding
        embedding = self.encoder(image)         #[batch_size, input_dim]
        labels = torch.unsqueeze(labels, 1)     #[batch_size, 1]

        #Forward pass through the Linear Layer
        outputs = self.model(embedding)         # [Batch_Size, Num_classes]

        if(self.activation == 'softmax'):
            probabilities = torch.softmax(outputs, dim=1)
            acc = self.calculate_acc(probabilities, labels, self.multilable)
            f1 = self.calculate_f1(probabilities, labels, self.multilable)

        elif(self.activation == 'sigmoid'):
            probabilities = torch.sigmoid(outputs)
            acc = self.calculate_acc(probabilities, labels, self.multilable)
            f1 = self.calculate_f1(probabilities, labels, self.multilable)

        #Calculate Loss
        #loss = self.criterion(outputs, labels.squeeze(1))
        loss = self.criterion(probabilities, labels.squeeze(1))

        #Predictions
        #preds = torch.softmax(outputs, dim=1)

        #acc = np.array(np.argmax(preds.cpu().detach().numpy(), axis=1) == labels.cpu().data.view(-1).numpy()).astype('int').sum().item() / embedding.size(0)

        #self.loss_sublist = np.append(self.loss_sublist, loss.cpu().detach().numpy())
        #self.acc_sublist = np.append(self.acc_sublist, acc, axis=0)
        
        #Logging metrics
        self.log('train_loss', loss, on_step=False, on_epoch=True, prog_bar=True, batch_size=self.batch_size)
        self.log('train_acc', acc, on_step=False, on_epoch=True, prog_bar=True, batch_size=self.batch_size)
        self.log('f1_score', f1, on_step=False, on_epoch=True, prog_bar=True, batch_size=self.batch_size)

        return {'acc':acc, 'loss':loss, 'f1_score':f1}

    def training_epoch_end(self, training_step_outputs):
        
        #print("Train acc: {} | Train Loss: {}".format(np.mean(self.acc_sublist), np.mean(self.loss_sublist)))
        
        print("\n")

    def validation_step(self, batch, batch_idx):
        image, labels = batch

        #Pass through the encoder to create the embedding
        embedding = self.encoder(image)                #[batch_size, input_dim]
        labels = torch.unsqueeze(labels, 1)            #[batch_size, 1]

        #Forward pass through the Linear Layer
        outputs = self.model(embedding)         # [Batch_Size, Num_classes]

        if(self.activation == 'softmax'):
            probabilities = torch.softmax(outputs, dim=1)
            acc = self.calculate_acc(probabilities, labels, self.multilable)
            f1 = self.calculate_f1(probabilities, labels, self.multilable)

        elif(self.activation == 'sigmoid'):
            probabilities = torch.sigmoid(outputs)
            acc = self.calculate_acc(probabilities, labels, self.multilable)
            f1 = self.calculate_f1(probabilities, labels, self.multilable)

        #Calculate Loss
        #loss = self.criterion(outputs, labels.squeeze(1))
        loss = self.criterion(probabilities, labels.squeeze(1))

        #preds = torch.softmax(outputs, dim=1)
        #acc = np.array(np.argmax(preds.cpu().detach().numpy(), axis=1) == labels.cpu().data.view(-1).numpy()).astype('int').sum().item() / embedding.size(0)

        #Logging metrics
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True, batch_size=self.batch_size)
        self.log('val_acc', acc, on_step=False, on_epoch=True, prog_bar=True, batch_size=self.batch_size)
        self.log('f1_score', f1, on_step=False, on_epoch=True, prog_bar=True, batch_size=self.batch_size)

        return {'acc':acc, 'loss':loss, 'f1_score':f1}

    def test_step(self, batch, batch_idx):
        image, labels = batch

        #Pass through the encoder to create the embedding
        embedding = self.encoder(image)                #[batch_size, input_dim]
        labels = torch.unsqueeze(labels, 1)            #[batch_size, 1]

        #Forward pass through the Linear Layer
        outputs = self.model(embedding)         # [Batch_Size, Num_classes]

        if(self.activation == 'softmax'):
            probabilities = torch.softmax(outputs, dim=1)
            acc = self.calculate_acc(probabilities, labels, self.multilable)
            f1 = self.calculate_f1(probabilities, labels, self.multilable)

        elif(self.activation == 'sigmoid'):
            probabilities = torch.sigmoid(outputs)
            acc = self.calculate_acc(probabilities, labels, self.multilable)
            f1 = self.calculate_f1(probabilities, labels, self.multilable)

        #Calculate Loss
        #loss = self.criterion(outputs, labels.squeeze(1))
        loss = self.criterion(probabilities, labels.squeeze(1))

        #preds = torch.softmax(outputs, dim=1)
        #acc = np.array(np.argmax(preds.cpu().detach().numpy(), axis=1) == labels.cpu().data.view(-1).numpy()).astype('int').sum().item() / embedding.size(0)

        #Logging metrics
        self.log('test_loss', loss, on_step=False, on_epoch=True, prog_bar=True, batch_size=self.batch_size)
        self.log('test_acc', acc, on_step=False, on_epoch=True, prog_bar=True, batch_size=self.batch_size)
        self.log('f1_score', f1, on_step=False, on_epoch=True, prog_bar=True, batch_size=self.batch_size)

        return {'acc':acc, 'loss':loss, 'f1_score':f1}

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)

        if(self.lr_scheduler == 'none'):
            return optimizer
        elif(self.lr_scheduler == 'reduce_plateau'):
            scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, min_lr=1e-6, patience=6, verbose=True)

        return {"optimizer": optimizer, 
                "lr_scheduler":{
                    "scheduler": scheduler,
                    "monitor": 'train_acc'
                }}








