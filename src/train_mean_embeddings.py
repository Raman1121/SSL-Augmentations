import os
import logging
import random
import numpy as np
import pandas as pd
from datetime import date

from utils import utils

import torch
from torch.utils.data import DataLoader
import torchvision.transforms as T

import pytorch_lightning as pl
from pytorch_lightning.loggers import CSVLogger

from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight

import albumentations as A
from albumentations.augmentations.transforms import *
from albumentations.augmentations.crops.transforms import *
from albumentations.augmentations.geometric.rotate import *
from albumentations.augmentations.geometric.transforms import *
from albumentations.augmentations.geometric.resize import Resize

from dataset import retinopathy_dataset, cancer_mnist_dataset, mura_dataset, chexpert_dataset
from model import supervised_model, mean_embedding_model

import argparse
import yaml
from pprint import pprint

##############################################################################################

parser = argparse.ArgumentParser(description='Hyper-parameters management')
parser.add_argument('--dataset', type=str, default='retinopathy', help='Dataset to use for training')
parser.add_argument('--experimental_run', type=bool, default=False, help='Experimental run (unit test)')
parser.add_argument('--num_runs', type=int, default=1, help='Number of runs for a dataset. Different augmentations will be randomly samples in each run.')
parser.add_argument('--train_mlp', type=bool, default=False, help='Train an MLP instead of a single layer')


args = parser.parse_args()

with open('../conf/config_integrated.yaml') as file:
        yaml_data = yaml.safe_load(file)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

if(device == torch.device(type='cpu')):
    GPUs = 0
elif(device == torch.device(type='cuda')):
    GPUs = 1

#RUN CONSTANTS
BATCH_SIZE = yaml_data['run']['batch_size']
DATASET = args.dataset
lr_rate = yaml_data['run']['lr_rate']
EPOCHS = yaml_data['run']['epochs']
EXPERIMENTAL_RUN = args.experimental_run
EMBEDDINGS_DIM = 2048
SUBSET = yaml_data['run']['subset']
AUTO_LR_FIND = yaml_data['run']['auto_lr_find']
NUM_RUNS = args.num_runs
SAVE_PLOTS = yaml_data['run']['save_plots']
EXPERIMENT = yaml_data['run']['experiment']
LOG_FOLDER = yaml_data['run']['log_folder']
DO_FINETUNE = yaml_data['run']['do_finetune']
ENCODER = yaml_data['run']['encoder']
LR_SCHEDULER = yaml_data['run']['lr_scheduler']
LOGGING = True
AUG_BIT_VECTOR = yaml_data['run']['aug_bit_vector']
TRAIN_MLP = args.train_mlp


#DATASET CONSTANTS
DATASET_ROOT_PATH = yaml_data['all_datasets'][DATASET]['root_path']
TRAIN_DF_PATH = yaml_data['all_datasets'][DATASET]['train_df_path']
VALIDATION_SPLIT = 0.3
SEED = 42
NUM_CLASSES = yaml_data['all_datasets'][DATASET]['num_classes']

#SAVING CONSTANTS
SAVED_MODELS_DIR = '../Saved_models'

if(EXPERIMENTAL_RUN):
    EPOCHS = 1
    AUTO_LR_FIND = False
    LOGGING = False

pprint(yaml_data)

if(LOGGING):

    #Save results to a text file
    filename = EXPERIMENT + '_' + DATASET + '_' + ENCODER + '_' + str(NUM_RUNS) + '.txt'
    f = open(os.path.join(LOG_FOLDER, filename), "w")
    f.write("EXPERIMENT DATE: {}".format(date.today()))
    f.write("\n")

    pprint(yaml_data, f)
    print("Dataset: {}".format(DATASET), f)
    print("SUBSET: {}".format(SUBSET), f)
    print("Number of Runs: {}".format(NUM_RUNS), f)
    print("Saved Models dir: {}".format(SAVED_MODELS_DIR), f)

crop_height = 224
crop_width = 224

aug_dict_labels = {
                   'CLAHE': CLAHE(),
                   'CJ': ColorJitter(),
                   'DS': Downscale(),
                   'EB': Emboss(),
                   'SSR': ShiftScaleRotate(),
                   'HF': HorizontalFlip(),
                   'VF': VerticalFlip(),
                   'IC': ImageCompression(),
                   'Rotate': Rotate(),
                   'INet_Norm':Normalize(),
                   'Perspective':Perspective()
                   }

main_df = pd.read_csv(TRAIN_DF_PATH)

if(AUG_BIT_VECTOR != None):

    assert len(AUG_BIT_VECTOR) == len(aug_dict_labels)

    aug_bit = AUG_BIT_VECTOR
    _selected_augs = utils.get_aug_from_vector(aug_dict_labels, AUG_BIT_VECTOR, get_labels=True)
    

    if(np.sum(AUG_BIT_VECTOR) == len(aug_dict_labels)):
        EXPERIMENT = 'mean_embedding_baseline_all_augs'
    elif(np.sum(AUG_BIT_VECTOR) == 0):
        EXPERIMENT = 'mean_embedding_baseline_no_augs'
    
else:
    _selected_augs, aug_bit = utils.gen_binomial_dict(aug_dict_labels)

if(TRAIN_MLP):
    EXPERIMENT = EXPERIMENT + '_mlp_'


print(_selected_augs)

raise SystemExit(0)