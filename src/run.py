import pprint
import argparse
import torch
import numpy as np
import pandas as pd
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

#from dataset import dataset
from utils import utils
from model import model

import yaml
from pprint import pprint


with open('config_train.yaml') as file:
    yaml_data = yaml.safe_load(file)

#################### DEFINE CONSTANTS ####################

ENCODER = yaml_data['model']['encoder']
BATCH_SIZE = yaml_data['training']['batch_size']    #Batch size of the dataset
NUM_AUG_SAMPLES = yaml_data['training']['num_aug_samples'] #Number of times to sample an augmentation
NUM_DUMMY_SAMPLES = yaml_data['dataset']['num_dummy_samples']
VERBOSE = yaml_data['training']['verbose']

USE_DATASET = yaml_data['dataset']['use_dataset']


#DIABETIC RETINOPATHY DATASET
DR_ROOT_PATH = yaml_data['DR_DATASET']['root_path']
DR_TRAIN_DF_PATH = yaml_data['DR_DATASET']['train_df_path']
DR_TEST_DF_PATH = yaml_data['DR_DATASET']['test_df_path']

#CHEXPERT DATASET
CHEX_ROOT_PATH = yaml_data['CHEX_DATASET']['root_path']
CHEX_TRAIN_DF_PATH = yaml_data['CHEX_DATASET']['train_df_path']
CHEX_VALID_DF_PATH = yaml_data['CHEX_DATASET']['valid_df_path']

##########################################################

aug_dict = {'RandomGrayscale': T.RandomGrayscale(p=0.2),
            'HorizontalFLip': T.RandomHorizontalFlip(),
            'ColorJitter': T.ColorJitter(0.4, 0.4, 0.4, 0.1)}

train_transforms = T.Compose([
    T.ToPILImage(),
    T.Resize((224,224)),
    T.ToTensor(),
    T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

test_transforms = T.Compose([
    T.ToPILImage(),
    T.Resize((224,224)),
    T.ToTensor()
])

##########################################################

if(USE_DATASET == 'diabetic_retinopathy'):
    train_dataset, test_dataset = utils.load_DR_dataset(yaml_data, train_transforms, test_transforms)

    print("\n")
    print("~~~~~~~~~~ Using DIABETIC RETINOPATHY DATASET ~~~~~~~~~~")
    print("Length of training dataset {}".format(train_dataset.__len__()))
    print("Length of test dataset {}".format(test_dataset.__len__()))

elif(USE_DATASET == 'dummy'):

    print("\n")
    print("~~~~~~~~~~ Using DUMMY DATASET ~~~~~~~~~~")
    train_dataset = utils.get_dummy_dataset(num_samples=NUM_DUMMY_SAMPLES)

elif(USE_DATASET == 'chexpert'):
    raise NotImplementedError("Not Implemented yet!!")

elif(USE_DATASET == 'cancer_mnist'):
    raise NotImplementedError("Not Implemented yet!!")

elif(USE_DATASET == 'mura'):
    raise NotImplementedError("Not Implemented yet!!")


##########################################################

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE)
utils.check_data_loader(train_loader)

encoder = model.Encoder(encoder=ENCODER)

final_dataset_embeddings = utils.run_one_aug(train_loader, encoder, aug_dict, NUM_DUMMY_SAMPLES, NUM_AUG_SAMPLES)

print(final_dataset_embeddings.size()) 
