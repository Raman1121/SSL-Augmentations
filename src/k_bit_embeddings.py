import os
import random
import numpy as np
import pandas as pd

import torch
from torch.utils.data import DataLoader
import torchvision.transforms as T

import pytorch_lightning as pl

from sklearn.model_selection import train_test_split

import albumentations as A
from albumentations.augmentations.transforms import *
from albumentations.augmentations.crops.transforms import *
from albumentations.augmentations.geometric.rotate import *
from albumentations.augmentations.geometric.resize import Resize


from dataset import retinopathy_dataset, cancer_mnist_dataset, mura_dataset, chexpert_dataset
from model import supervised_model

import argparse
import yaml

##############################################################################################

parser = argparse.ArgumentParser(description='Hyper-parameters management')
parser.add_argument('--dataset', type=str, default='retinopathy', help='Dataset to use for training')
parser.add_argument('--experimental_run', type=bool, default=False, help='Experimental run (unit test)')

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



transform_prob = 1
#crop_height = int(0.7*224) #Assuming all images would be 224x224
#crop_width = int(0.7*224)  #Assuming all images would be 224x224

crop_height = 224
crop_width = 224

aug_list = [CLAHE(p=transform_prob), ColorJitter(p=transform_prob), 
            Downscale(p=transform_prob), Emboss(p=transform_prob), Flip(p=transform_prob),
            HorizontalFlip(p=transform_prob), VerticalFlip(p=transform_prob), ImageCompression(p=transform_prob), 
            Rotate(p=transform_prob)]

#CenterCrop(crop_height, crop_width, p=transform_prob),

print("Number of augmentations: ", len(aug_list))
randomly_selected_augs = random.sample(aug_list, int(0.7*len(aug_list)))

print("{} Augmentations selected: {} \n".format(len(randomly_selected_augs), [i for i in randomly_selected_augs]))

#TODO: Add required basic transforms here.
randomly_selected_augs = [Resize(224, 224)] + randomly_selected_augs

train_transform = A.Compose(randomly_selected_augs)


#Load the train set
main_df = pd.read_csv(TRAIN_DF_PATH)

##################################### DATASETS #######################################

train_dataset = None
test_dataset = None
val_dataset = None

if(DATASET == 'retinopathy'):

    '''
    Preparing the Diabetic Retinopathy dataset
    '''
    
    #Load the test set for DR dataset
    TEST_DF_PATH = yaml_data['all_datasets'][DATASET]['test_df_path']
    test_df = pd.read_csv(TEST_DF_PATH)

    TRAIN_CAT_LABELS = yaml_data['all_datasets'][DATASET]['train_cat_labels']
    VAL_CAT_LABELS = yaml_data['all_datasets'][DATASET]['val_cat_labels']
    TEST_CAT_LABELS = yaml_data['all_datasets'][DATASET]['test_cat_labels']

    main_df['image'] = main_df['image'].apply(lambda x: str(DATASET_ROOT_PATH+'final_train/train/'+x))
    test_df['image'] = test_df['image'].apply(lambda x: str(DATASET_ROOT_PATH+'final_test/test/'+x))

    train_dataset = retinopathy_dataset.RetinopathyDataset(df=main_df, cat_labels_to_include=TRAIN_CAT_LABELS, 
                                                        transforms=train_transform, subset=SUBSET)

    test_dataset = retinopathy_dataset.RetinopathyDataset(df=test_df, cat_labels_to_include=TEST_CAT_LABELS, 
                                                        transforms=train_transform, subset=SUBSET)

elif(DATASET == 'cancer_mnist'):

    '''
    Preparing the Cancer MNIST dataset
    '''

    #NOTE: Test data for this dataset has not been provided!

    # Creating training and test splits
    train_df, test_df = train_test_split(main_df, test_size=VALIDATION_SPLIT,
                                random_state=SEED)
    train_df = train_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)

    train_dataset = cancer_mnist_dataset.CancerMNISTDataset(df=train_df, transforms=train_transform, 
                                                            subset=SUBSET)

    test_dataset = cancer_mnist_dataset.CancerMNISTDataset(df=test_df, transforms=train_transform, 
                                                            subset=SUBSET)

elif(DATASET == 'chexpert'):
    '''
    Preparing the CheXpert dataset
    '''
    #NOTE: Test data for this dataset has not been provided!

    VAL_DF_PATH = yaml_data['all_datasets'][DATASET]['val_df_path']
    val_df = pd.read_csv(VAL_DF_PATH)

    # Creating training and test splits
    train_df, test_df = train_test_split(main_df, test_size=VALIDATION_SPLIT,
                                random_state=SEED)

    train_df = train_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)

    
    train_dataset = chexpert_dataset.ChexpertDataset(df=train_df, transforms=train_transform, 
                                                    subset=SUBSET)

    val_dataset = chexpert_dataset.ChexpertDataset(df=val_df, transforms=train_transform,
                                                    subset=SUBSET)

    test_dataset = chexpert_dataset.ChexpertDataset(df=test_df, transforms=train_transform, 
                                                    subset=SUBSET)

elif(DATASET == 'mura'):
    '''
    Preparing the MURA dataset
    '''

    #NOTE: Test data for this dataset has not been provided!

    VAL_DF_PATH = yaml_data['all_datasets'][DATASET]['val_df_path']
    val_df = pd.read_csv(VAL_DF_PATH)

    # Creating training and test splits
    train_df, test_df = train_test_split(main_df, test_size=VALIDATION_SPLIT,
                                random_state=SEED)

    train_df = train_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)

    train_dataset = mura_dataset.MuraDataset(df=train_df, transforms=train_transform, 
                                                            subset=SUBSET)

    val_dataset = mura_dataset.MuraDataset(df=val_df, transforms=train_transform,
                                                            subset=SUBSET)

    test_dataset = mura_dataset.MuraDataset(df=test_df, transforms=train_transform, 
                                                            subset=SUBSET)

######################################################################################


#Creating Data Loaders
if(train_dataset != None):
    train_image_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=12)
if(test_dataset != None):
    test_image_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=12)
if(val_dataset != None):
    val_image_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=12)

supervised_model = supervised_model.SupervisedModel(encoder='resnet50_supervised', batch_size = BATCH_SIZE, num_classes=NUM_CLASSES,
                                                    lr_rate=lr_rate, lr_scheduler='none')

trainer = pl.Trainer(gpus=GPUs, 
                    max_epochs=EPOCHS)

if(AUTO_LR_FIND):
    lr_finder = trainer.tuner.lr_find(supervised_model, train_image_loader)
    new_lr = lr_finder.suggestion()
    print("New suggested learning rate is: ", new_lr)
    supervised_model.hparams.learning_rate = new_lr

if(val_dataset == None):

    print("Validation dataset not provided for {} dataset".format(DATASET))
    #Providing data loader for only the train set in the fit method.
    trainer.fit(supervised_model, train_image_loader)

elif(val_dataset != None):

    #Providing data loader for both the train and val set in the fit method.
    trainer.fit(supervised_model, train_image_loader, val_image_loader)

if(test_dataset != None):
    trainer.test(dataloaders=test_image_loader)
else:
    print("Test data not provided for {} dataset hence, skipping testing.".format(DATASET))