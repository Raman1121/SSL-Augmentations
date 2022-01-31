import pprint
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

aug_list = [T.RandomGrayscale(p=0.2), T.RandomHorizontalFlip(),
             T.ColorJitter(0.4, 0.4, 0.4, 0.1)]

'''
* Feed a dataset of images and obtain embeddings for all images. 
* Final output for a dataset should have size [num_samples, embedding_size]
* 'embedding_size' depends on the encoder used. [ResNet, SimCLR = 2048, ViTs = 768]
'''

BATCH_SIZE = 10

ds = dataset.get_dummy_dataset()
dl = DataLoader(ds, batch_size=BATCH_SIZE)
encoder = model.Encoder(encoder='vit_base_patch32_224_in21k')

all_embeddings = torch.tensor([])
final_dataset_embeddings = torch.tensor([])

for x, y in dl:
    
    #print(" ############ BATCH ############")

    #For each image, apply each augmentation
    for i in range(x.size()[0]):
        all_embed_img = torch.tensor([])    #A tensor to hold embeddings of all augmentations of an image
        #print('Image: ', i+1)

        for aug in aug_list:
            #print("Augmentation: ", aug)
            preprocess = T.Compose([T.ToPILImage(), aug, T.ToTensor()])
            aug_img = preprocess(x[i])
            embedding = encoder(aug_img.unsqueeze(0))

            all_embed_img = torch.cat((all_embed_img,
                                    embedding), 0)

        all_embeddings = torch.cat((all_embeddings,
                                    torch.mean(all_embed_img, 0)), 0)

        #print('\n')

final_dataset_embeddings = torch.reshape(all_embeddings, (ds.num_samples, embedding.size()[1]))

print(final_dataset_embeddings.size()) 