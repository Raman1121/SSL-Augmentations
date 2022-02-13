import pprint
import argparse
import torch
import numpy as np
import urllib
from PIL import Image

import timm
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform


from torch import nn
import torchvision
import torchvision.models as models
import torchvision.transforms as T
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader

from pl_bolts.models.self_supervised import SimCLR
from pytorch_lightning.core.lightning import LightningModule

from pytorch_lightning import Trainer
from pl_bolts.datasets import DummyDataset

from dataset import dataset
from model import model

import yaml
from pprint import pprint

# parser = argparse.ArgumentParser(description='Hyper-parameters management')

# parser.add_argument('--num_samples', type=int, default=5, help='Number of samples in a dataset')
# parser.add_argument('--batch_size', type=int, default=64, help='Batch Size')
# parser.add_argument('--num_aug_samples', type=int, default=10, help='Number of times you want to sample an augmentation')
# parser.add_argument('--encoder', type=str, default='resnet50_supervised', help='Encoder Model')

with open('config_train.yaml') as file:
    yaml_data = yaml.safe_load(file)

#################### DEFINE CONSTANTS ####################

ENCODER = yaml_data['model']['encoder']
BATCH_SIZE = yaml_data['training']['batch_size']    #Batch size of the dataset
NUM_AUG_SAMPLES = yaml_data['training']['num_aug_samples'] #Number of times to sample an augmentation
NUM_DUMMY_SAMPLES = yaml_data['dataset']['num_samples']
VERBOSE = yaml_data['training']['verbose']

##########################################################

aug_dict = {'RandomGrayscale': T.RandomGrayscale(p=0.2),
            'HorizontalFLip': T.RandomHorizontalFlip(),
            'ColorJitter':T.ColorJitter(0.4, 0.4, 0.4, 0.1)}


'''
* Feed a dataset of images and obtain embeddings for all images. 
* Final output for a dataset should have size [num_samples, embedding_size]
* 'embedding_size' depends on the encoder used. [ResNet, SimCLR = 2048, ViTs = 768]
'''
                       

ds = dataset.get_dummy_dataset(num_samples=NUM_DUMMY_SAMPLES)
dl = DataLoader(ds, batch_size=BATCH_SIZE)
encoder = model.Encoder(encoder=ENCODER)

all_embeddings = torch.tensor([])   #A tensor to hold mean embeddings of all images in a dataset
final_dataset_embeddings = torch.tensor([]) #A tensor holding mean embeddings for all images in a dataset but reshaped to correct dimensions

for x, y in dl:
    
    '''
    LOOP NUMBER 1
    For each image:
    [Selecting an image from the dataset]

    '''
    for i in range(x.size()[0]):
        all_embed_img = torch.tensor([])    #A tensor to hold embeddings of all augmentations of an image

        '''
        LOOP NUMBER 2
        For each augmentation:
        [Selecting an augmentation from the set of all augmentations]

        '''
        for j in range(len(aug_dict)):
            aug = list(aug_dict.values())[j]
            all_embed_aug = torch.tensor([])    #A tensor to hold all embeddings for different samples of a particular augmentation

            #print("Augmentation: ", aug)

            '''
            LOOP NUMBER 3
            FOR EACH SAMPLE OF AN AUGMENTATION
            [Sampling an augmentation]
            '''
            for k in range(NUM_AUG_SAMPLES):
                
                #print("Sample number for this augmentation: ", k)
                preprocess = T.Compose([T.ToPILImage(), aug, T.ToTensor()])
                aug_img = preprocess(x[i])
                embedding = encoder(aug_img.unsqueeze(0))

                all_embed_aug = torch.cat((all_embed_aug, embedding), 0)

                #print("Size of the tensor holding all samples for this augmentation: ", all_embed_aug.size())
            
            all_embed_aug = torch.mean(all_embed_aug, 0)    #Mean of different samples of the same augmentation [1, embedding_size]

            all_embed_img = torch.cat((all_embed_img,       #Adding the mean embedding obtained via different samples of an augmentation.
                                        all_embed_aug), 0)

        all_embed_img = torch.reshape(all_embed_img, (len(aug_dict), embedding.size()[1]))   #[number of augmentations, embedding_size]
        all_embed_img = torch.mean(all_embed_img, 0) #Final embeddings generated after multiple samples of all augmentations for an image [1, embedding_size]
        
        all_embeddings = torch.cat((all_embeddings,         #Adding mean embedding for an image to the list of all embeddings for a dataset.
                                    all_embed_img), 0)

    print("SHAPE OF ALL EMBEDDINGS FOR ALL IMAGES", all_embeddings.size()) #[num_samples, 768]

    print('\n')

final_dataset_embeddings = torch.reshape(all_embeddings, (ds.num_samples, embedding.size()[1])) #Final reshaping for all images in a dataset.

print(final_dataset_embeddings.size()) 