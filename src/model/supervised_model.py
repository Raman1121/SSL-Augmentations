import torch
from torch import nn

import timm
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform

from torch.optim.lr_scheduler import CyclicLR, CosineAnnealingLR, ReduceLROnPlateau
from torch.optim import Adam
from torch.nn.functional import cross_entropy

import numpy as np
from sklearn.metrics import accuracy_score
import torchvision.models as models
from pl_bolts.models.self_supervised import SimCLR
from pytorch_lightning.core.lightning import LightningModule


class SupervisedModel(LightningModule):
    def __init__(self, num_classes, batch_size, encoder='resnet50_supervised', lr_rate=0.001, 
                lr_scheduler='none', do_finetune=True, activation='softmax', criterion='cross_entropy', 
                multilable=False):

        super().__init__()
        
        self.encoder = encoder
        self.num_classes = num_classes
        self.learning_rate = lr_rate
        self.lr_scheduler = lr_scheduler
        self.batch_size = batch_size
        self.activation = activation
        self.multilable = multilable
        #self.criterion = nn.BCELoss()
        
        if(criterion == 'cross_entropy'):
            #self.criterion = nn.functional.cross_entropy()  #This combines log_softmax with cross_entropy
            self.criterion = nn.CrossEntropyLoss()
        elif(criterion == 'bce'):
            #self.criterion = nn.BCEWithLogitsLoss()         #This combines sigmoid with BCE.
            self.criterion = nn.BCELoss()
        

        if(self.encoder == 'resnet50_supervised'):
            backbone = models.resnet50(pretrained=True)
            num_filters = backbone.fc.in_features
            layers = list(backbone.children())[:-1]
            self.feature_extractor = nn.Sequential(*layers)
            self.classifier = nn.Linear(num_filters, self.num_classes)

            # if(activation == 'softmax'):
            #     self.classifier = nn.Sequential(nn.Linear(num_filters, self.num_classes), nn.Softmax())
            # elif(activation == 'sigmoid'):
            #     self.classifier = nn.Sequential(nn.Linear(num_filters, self.num_classes), nn.Sigmoid())

            if(do_finetune):

                self.feature_extractor.eval()
                
                for param in self.feature_extractor.parameters():
                    param.requires_grad = False

    # def forward(self, x):
    #     # self.feature_extractor.eval()
    #     # with torch.no_grad():
    #     #     representations = self.feature_extractor(x).flatten(1)

    #     # x = self.classifier(representations)

    #     x = self.feature_extractor(x)
    #     x = x.view(x.size(0), -1)
    #     x = self.classifier(x)

    #     return x

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
            acc = np.array(np.argmax(probs.cpu().detach().numpy(), axis=1) == true_labels.cpu().data.view(-1).numpy()).astype('int').sum().item() / probs.size(0)
            
        return acc

    def training_step(self, batch, batch_idx):

        #total_acc = 0
        
        images, labels = batch

        representations = self.feature_extractor(images).flatten(1)

        logits = self.classifier(representations)  #LOGITS (Unnormalized)

        #Convert Logits to probabilities
        if(self.activation == 'softmax'):
            probabilities = torch.softmax(logits, dim=1)    #Normalized
            #acc = np.array(np.argmax(probabilities.cpu().detach().numpy(), axis=1) == labels.cpu().data.view(-1).numpy()).astype('int').sum().item() / representations.size(0)
            acc = self.calculate_acc(probabilities, labels, self.multilable)
            
        elif(self.activation == 'sigmoid'):
            probabilities = torch.sigmoid(logits)
            acc = self.calculate_acc(probabilities, labels, self.multilable)
            

        #print("representations: ", representations.shape)
        #print("logits: ", logits.shape)
        #print("logits max: ", torch.max(logits, dim=1))
        #print("probabilities: ", probabilities)
        #print("labels: ", labels)
                    
        #loss = self.criterion(torch.max(logits, dim=1).values, labels)  #We are using outputs here since our loss function applies final activation automatically.

        #Calculate loss using probabilities
        loss = self.criterion(probabilities, labels)

        #Logging metrics
        self.log('train_loss', loss, on_step=False, on_epoch=True, prog_bar=True, batch_size=self.batch_size)
        self.log('train_acc', acc, on_step=False, on_epoch=True, prog_bar=True, batch_size=self.batch_size)

        return {'acc':acc, 'loss':loss}


    def validation_step(self, batch, batch_idx):
        
        images, labels = batch

        representations = self.feature_extractor(images).flatten(1)

        logits = self.classifier(representations)  #LOGITS (Unnormalized)

        #Convert Logits to probabilities
        if(self.activation == 'softmax'):
            probabilities = torch.softmax(logits, dim=1)    #Normalized
            #acc = np.array(np.argmax(probabilities.cpu().detach().numpy(), axis=1) == labels.cpu().data.view(-1).numpy()).astype('int').sum().item() / representations.size(0)
            acc = self.calculate_acc(probabilities, labels, self.multilable)
            
        elif(self.activation == 'sigmoid'):
            probabilities = torch.sigmoid(logits)
            acc = self.calculate_acc(probabilities, labels, self.multilable)

        #Calculate loss using probabilities
        loss = self.criterion(probabilities, labels)

        #Logging metrics
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True, batch_size=self.batch_size)
        self.log('val_acc', acc, on_step=False, on_epoch=True, prog_bar=True, batch_size=self.batch_size)

        return loss
        


    def test_step(self, batch, batch_idx):
        
        images, labels = batch

        representations = self.feature_extractor(images).flatten(1)

        logits = self.classifier(representations)  #LOGITS (Unnormalized)

        #Convert Logits to probabilities
        if(self.activation == 'softmax'):
            probabilities = torch.softmax(logits, dim=1)    #Normalized
            #acc = np.array(np.argmax(probabilities.cpu().detach().numpy(), axis=1) == labels.cpu().data.view(-1).numpy()).astype('int').sum().item() / representations.size(0)
            acc = self.calculate_acc(probabilities, labels, self.multilable)
            
        elif(self.activation == 'sigmoid'):
            probabilities = torch.sigmoid(logits)
            acc = self.calculate_acc(probabilities, labels, self.multilable)

        #Calculate loss using probabilities
        loss = self.criterion(probabilities, labels)

        #Logging metrics
        self.log('test_loss', loss, on_step=False, on_epoch=True, prog_bar=True, batch_size=self.batch_size)
        self.log('test_acc', acc, on_step=False, on_epoch=True, prog_bar=True, batch_size=self.batch_size)

        return loss
        

    
        
    def configure_optimizers(self):
        #optimizer = torch.optim.Adam(self.classifier.parameters(), lr=self.learning_rate)
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)

        if(self.lr_scheduler == 'none'):
            return optimizer
        elif(self.lr_scheduler == 'reduce_plateau'):
            scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, min_lr=1e-6, patience=6, verbose=True)

        return {"optimizer": optimizer, 
                "lr_scheduler":{
                    "scheduler": scheduler,
                    "monitor": 'train_loss'
                }}

    


