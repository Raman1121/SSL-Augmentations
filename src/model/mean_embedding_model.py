import torch
from torch import nn

from torch.optim.lr_scheduler import CyclicLR, CosineAnnealingLR, ReduceLROnPlateau
from torch.optim import Adam
from torch.nn.functional import cross_entropy

import numpy as np
from sklearn.metrics import accuracy_score, f1_score
import torchvision.models as models
from pl_bolts.models.self_supervised import SimCLR
from pytorch_lightning.core.lightning import LightningModule

import albumentations as A
import torchvision.transforms as T
from albumentations.augmentations.transforms import *
from albumentations.augmentations.crops.transforms import *
from albumentations.augmentations.geometric.rotate import *
from albumentations.augmentations.geometric.transforms import *
from albumentations.augmentations.geometric.resize import Resize

class MeanEmbeddingModel(LightningModule):
    def __init__(self, num_classes, batch_size, class_weights, encoder='resnet50_supervised', lr_rate=0.001, 
                lr_scheduler='none', do_finetune=True, train_mlp=False, activation='softmax', criterion='cross_entropy', 
                multilable=False, aug_list=None, new_aug_dict=None, k=10):

        super().__init__()

        self.num_classes = num_classes
        self.batch_size = batch_size
        self.class_weights = class_weights
        self.encoder = encoder
        self.learning_rate = lr_rate
        self.lr_scheduler = lr_scheduler
        self.activation = activation
        self.multilable = multilable
        self.aug_list = aug_list            #List of augmentation labels
        self.new_aug_dict = new_aug_dict
        self.k = k

        assert self.aug_list != None
        assert self.new_aug_dict != None

        if(criterion == 'cross_entropy'):
            #self.criterion = nn.functional.cross_entropy()  #This combines log_softmax with cross_entropy
            self.criterion = nn.CrossEntropyLoss(weight=self.class_weights)
        elif(criterion == 'bce'):
            #self.criterion = nn.BCEWithLogitsLoss()         #This combines sigmoid with BCE.
            self.criterion = nn.BCELoss()
        

        if(self.encoder == 'resnet50'):
            backbone = models.resnet50(pretrained=True)
            num_filters = backbone.fc.in_features
            layers = list(backbone.children())[:-1]
            self.feature_extractor = nn.Sequential(*layers)

            if(train_mlp):
                self.classifier = nn.Sequential(nn.Linear(num_filters, 512),
                                                 nn.ReLU(),
                                                 nn.Linear(512, 128),
                                                 nn.ReLU(),
                                                 nn.Linear(128, self.num_classes))
            else:
                self.classifier = nn.Linear(num_filters, self.num_classes)            

        elif(self.encoder == 'vit_patch16'):
            self.feature_extractor = timm.create_model('vit_base_patch16_224_in21k', pretrained=True, num_classes=0)
            config = resolve_data_config({}, model=self.feature_extractor)
            transform = create_transform(**config)

            if(train_mlp):
                self.classifier = nn.Sequential(nn.Linear(768, 256),
                                                 nn.ReLU(),
                                                 nn.Linear(256, 128),
                                                 nn.ReLU(),
                                                 nn.Linear(128, self.num_classes))
            else:
                self.classifier = nn.Linear(768, self.num_classes)
            
        elif(self.encoder == 'simclr'):
            weight_path = 'https://pl-bolts-weights.s3.us-east-2.amazonaws.com/simclr/bolts_simclr_imagenet/simclr_imagenet.ckpt'
            simclr = SimCLR.load_from_checkpoint(weight_path, strict=False)
            self.feature_extractor = simclr.encoder

            if(train_mlp):
                self.classifier = nn.Sequential(nn.Linear(2048, 512),
                                                 nn.ReLU(),
                                                 nn.Linear(512, 128),
                                                 nn.ReLU(),
                                                 nn.Linear(128, self.num_classes))
            else:
                self.classifier = nn.Linear(2048, self.num_classes)

        #Check for finetuning
        if(not do_finetune):

            self.feature_extractor.eval()
            
            for param in self.feature_extractor.parameters():
                param.requires_grad = False

    def get_aug_object(self, aug_label):
        
        return self.new_aug_dict[aug_label]


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

    def get_mean_embeddings(self, images):

        
        batch_images_tensor = torch.Tensor([])  # A tensor to hold all the images in a batch
        batch_images_tensor = batch_images_tensor.cuda()

        for i in range(images.shape[0]):

            '''
            LOOP NUMBER 1
            For each image:
            [Selecting an image from a batch]
            '''

            all_embed_img = torch.tensor([])    #A tensor to hold embeddings of all augmentations of an image
            all_embed_img = all_embed_img.cuda()

            _img = images[i]

            #Convert to channel_last format for albumentations
            _img = np.asarray(T.ToPILImage()(_img))

            #_img = _img.numpy()

            for _aug in self.aug_list:

                '''
                LOOP NUMBER 2
                For each augmentation:
                [Selecting an augmentation from the set of all augmentations]
                '''

                all_embed_aug = torch.tensor([])    #A tensor to hold all embeddings for different samples of a particular augmentation
                all_embed_aug = all_embed_aug.cuda()

                for _k in range(self.k):

                    '''
                    LOOP NUMBER 3
                    FOR EACH SAMPLE OF AN AUGMENTATION
                    [Sampling an augmentation]
                    '''


                    sampled_aug = self.get_aug_object(_aug)  #New augmentation sampled with different random params
                    transform = A.Compose([Resize(224, 224), sampled_aug])
                    transformed_img = transform(image = _img)["image"]

                    #print(type(transformed_img))
                    #print(transformed_img.shape)        # [224, 224, 3]

                    #Convert this image to a float tensor
                    transformed_img = torch.from_numpy(transformed_img)
                    transformed_img = transformed_img.float()

                    #Convert to channel_first format here
                    transformed_img = transformed_img.permute(2, 0, 1)      # [3, 224, 224]

                    #Expand dimension
                    transformed_img = transformed_img[None, :]
                    transformed_img = transformed_img.cuda()
                    #print(transformed_img.shape)          

                    #Pass each transformed input through the model
                    if(self.encoder == 'resnet50'):
                        representations = self.feature_extractor(transformed_img).flatten(1)
                    elif(self.encoder == 'vit_patch16'):
                        representations = self.feature_extractor(transformed_img)
                    elif(self.encoder == 'simclr'):
                        representations = self.feature_extractor(transformed_img)[0]

                    #Append the Representations and take mean
                    #all_embed_img = torch.cat((all_embed_img, representations), 0)
                    #print("representations: ", representations.shape[1])
                    all_embed_aug = torch.cat((all_embed_aug, representations), 0)

                all_embed_aug = torch.mean(all_embed_aug, 0)    #Mean of different samples of the same augmentation [1, 2048]
            
                all_embed_img = torch.cat((all_embed_img,           #Adding the mean embedding obtained via different samples of an augmentation.
                                       all_embed_aug), 0)
                #all_embed_img = torch.mean(all_embed_img, 0)  #[1, 2048]
                #print("Shape here: ", all_embed_img.shape)

            all_embed_img = torch.reshape(all_embed_img, (len(self.aug_list), representations.size()[1]))   #[number of augmentations, embedding_size]
            
            all_embed_img = torch.mean(all_embed_img, 0) #Final embeddings generated after multiple samples of all augmentations for an image [1, embedding_size]
            #print(all_embed_img.shape)
            batch_images_tensor = torch.cat((batch_images_tensor, all_embed_img), 0)    
        
        batch_images_tensor = torch.reshape(batch_images_tensor, (images.shape[0], representations.shape[1]))       #Shape: [batch_size, 2048]
        assert batch_images_tensor.shape == torch.Size([images.shape[0], representations.shape[1]])

        return batch_images_tensor


    def training_step(self, batch, batch_idx):
        
        images, labels = batch

        #print(images.device)

        images = images.cuda()
        labels = labels.cuda()

        #print(images.device)
        
        batch_images_tensor = self.get_mean_embeddings(images)
        logits = self.classifier(batch_images_tensor)  #LOGITS (Unnormalized)

        #Convert Logits to probabilities
        if(self.activation == 'softmax'):
            probabilities = torch.softmax(logits, dim=1)    #Normalized
            #acc = np.array(np.argmax(probabilities.cpu().detach().numpy(), axis=1) == labels.cpu().data.view(-1).numpy()).astype('int').sum().item() / representations.size(0)
            acc = self.calculate_acc(probabilities, labels, self.multilable)
            f1 = self.calculate_f1(probabilities, labels, self.multilable)

        elif(self.activation == 'sigmoid'):
            probabilities = torch.sigmoid(logits)
            acc = self.calculate_acc(probabilities, labels, self.multilable)
            f1 = self.calculate_f1(probabilities, labels, self.multilable)

        #print("Probabilities Shape: ", probabilities.shape)
        #print("Labels Shape: ", labels.shape)

        #Calculate loss using probabilities
        loss = self.criterion(probabilities, labels)

        #Logging metrics
        self.log('train_loss', loss, on_step=False, on_epoch=True, prog_bar=True, batch_size=self.batch_size)
        self.log('train_acc', acc, on_step=False, on_epoch=True, prog_bar=True, batch_size=self.batch_size)
        self.log('f1_score', f1, on_step=False, on_epoch=True, prog_bar=True, batch_size=self.batch_size)

        return {'acc':acc, 'loss':loss, 'f1_score':f1}

    
    def validation_step(self, batch, batch_idx):
        images, labels = batch

        images = images.cuda()
        labels = labels.cuda()
        
        batch_images_tensor = self.get_mean_embeddings(images)
        logits = self.classifier(batch_images_tensor)  #LOGITS (Unnormalized)

        #Convert Logits to probabilities
        if(self.activation == 'softmax'):
            probabilities = torch.softmax(logits, dim=1)    #Normalized
            #acc = np.array(np.argmax(probabilities.cpu().detach().numpy(), axis=1) == labels.cpu().data.view(-1).numpy()).astype('int').sum().item() / representations.size(0)
            acc = self.calculate_acc(probabilities, labels, self.multilable)
            f1 = self.calculate_f1(probabilities, labels, self.multilable)

        elif(self.activation == 'sigmoid'):
            probabilities = torch.sigmoid(logits)
            acc = self.calculate_acc(probabilities, labels, self.multilable)
            f1 = self.calculate_f1(probabilities, labels, self.multilable)

        #Calculate loss using probabilities
        loss = self.criterion(probabilities, labels)

        #Logging metrics
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True, batch_size=self.batch_size)
        self.log('val_acc', acc, on_step=False, on_epoch=True, prog_bar=True, batch_size=self.batch_size)
        self.log('f1_score', f1, on_step=False, on_epoch=True, prog_bar=True, batch_size=self.batch_size)

        return {'acc':acc, 'loss':loss, 'f1_score':f1}


    def test_step(self, batch, batch_idx):
        images, labels = batch

        images = images.cuda()
        labels = labels.cuda()

        print(images.dtype)
        print(labels.dtype)

        batch_images_tensor = self.get_mean_embeddings(images)
        logits = self.classifier(batch_images_tensor)  #LOGITS (Unnormalized)

        #Convert Logits to probabilities
        if(self.activation == 'softmax'):
            probabilities = torch.softmax(logits, dim=1)    #Normalized
            #acc = np.array(np.argmax(probabilities.cpu().detach().numpy(), axis=1) == labels.cpu().data.view(-1).numpy()).astype('int').sum().item() / representations.size(0)
            acc = self.calculate_acc(probabilities, labels, self.multilable)
            f1 = self.calculate_f1(probabilities, labels, self.multilable)

        elif(self.activation == 'sigmoid'):
            probabilities = torch.sigmoid(logits)
            acc = self.calculate_acc(probabilities, labels, self.multilable)
            f1 = self.calculate_f1(probabilities, labels, self.multilable)

        #Calculate loss using probabilities
        loss = self.criterion(probabilities, labels)

        #Logging metrics
        self.log('test_loss', loss, on_step=False, on_epoch=True, prog_bar=True, batch_size=self.batch_size)
        self.log('test_acc', acc, on_step=False, on_epoch=True, prog_bar=True, batch_size=self.batch_size)
        self.log('f1_score', f1, on_step=False, on_epoch=True, prog_bar=True, batch_size=self.batch_size)

        return {'acc':acc, 'loss':loss, 'f1_score':f1}



    def configure_optimizers(self):
        #optimizer = torch.optim.Adam(self.classifier.parameters(), lr=self.learning_rate)
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate, weight_decay=0.0008)

        if(self.lr_scheduler == 'none'):
            return optimizer
        elif(self.lr_scheduler == 'reduce_plateau'):
            scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, min_lr=1e-6, patience=6, verbose=True)
        elif(self.lr_scheduler == 'cyclic'):
            scheduler = CyclicLR(optimizer, base_lr=self.learning_rate, max_lr=1e-1, cycle_momentum=False, verbose=True)
        elif(self.lr_scheduler == 'cosine'):
            scheduler = CosineAnnealingLR(optimizer, T_max=1000, verbose=True)

        return {"optimizer": optimizer, 
                "lr_scheduler":{
                    "scheduler": scheduler,
                    "monitor": 'train_loss'
                }}








    