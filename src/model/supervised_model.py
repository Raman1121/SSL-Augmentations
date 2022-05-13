import torch
from torch import nn

import timm
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform

from torch.optim.lr_scheduler import CyclicLR, CosineAnnealingLR, ReduceLROnPlateau
from torch.optim import Adam
from torch.nn.functional import cross_entropy

import numpy as np
import torchvision.models as models
from pl_bolts.models.self_supervised import SimCLR
from pytorch_lightning.core.lightning import LightningModule


class SupervisedModel(LightningModule):
    def __init__(self, num_classes, batch_size, encoder='resnet50_supervised', lr_rate=0.001, lr_scheduler='none'):
        super().__init__()
        
        self.encoder = encoder
        self.num_classes = num_classes
        self.criterion = nn.CrossEntropyLoss()
        self.learning_rate = lr_rate
        self.lr_scheduler = lr_scheduler
        self.batch_size = batch_size
        #backbone = models.resnet50(pretrained=True)

        if(self.encoder == 'resnet50_supervised'):
            backbone = models.resnet50(pretrained=True)
            num_filters = backbone.fc.in_features
            layers = list(backbone.children())[:-1]
            self.feature_extractor = nn.Sequential(*layers)
            self.classifier = nn.Linear(num_filters, self.num_classes)

    def forward(self, x):
        self.feature_extractor.eval()
        with torch.no_grad():
            representations = self.feature_extractor(x).flatten(1)

        x = self.classifier(representations)

        return x

    def training_step(self, batch, batch_idx):
        images, labels = batch

        representations = self.feature_extractor(images).flatten(1)

        outputs = self.classifier(representations)

        #print("representations: ", representations.shape)
        #print("outputs: ", outputs.shape)
        #print("labels: ", labels.shape)

        loss = self.criterion(outputs, labels)

        #Predictions
        preds = torch.softmax(outputs, dim=1)

        acc = np.array(np.argmax(preds.cpu().detach().numpy(), axis=1) == labels.cpu().data.view(-1).numpy()).astype('int').sum().item() / representations.size(0)

        #Logging metrics
        self.log('train_loss', loss, on_step=False, on_epoch=True, prog_bar=True, batch_size=self.batch_size)
        self.log('train_acc', acc, on_step=False, on_epoch=True, prog_bar=True, batch_size=self.batch_size)

        return loss

    def test_step(self, batch, batch_idx):
        images, labels = batch

        representations = self.feature_extractor(images).flatten(1)

        outputs = self.classifier(representations)

        loss = self.criterion(outputs, labels)

        preds = torch.softmax(outputs, dim=1)

        acc = np.array(np.argmax(preds.cpu().detach().numpy(), axis=1) == labels.cpu().data.view(-1).numpy()).astype('int').sum().item() / representations.size(0)

        #Logging metrics
        self.log('test_loss', loss, on_step=False, on_epoch=True, prog_bar=True, batch_size=self.batch_size)
        self.log('test_acc', acc, on_step=False, on_epoch=True, prog_bar=True, batch_size=self.batch_size)

        return loss


    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.classifier.parameters(), lr=self.learning_rate)

        if(self.lr_scheduler == 'none'):
            return optimizer
        elif(self.lr_scheduler == 'reduce_plateau'):
            scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, min_lr=1e-6, patience=6, verbose=True)

        return {"optimizer": optimizer, 
                "lr_scheduler":{
                    "scheduler": scheduler,
                    "monitor": 'train_acc'
                }}

    


