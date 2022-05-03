import torch
import os
from dataset import retinopathy_dataset
import pandas as pd
import matplotlib.pyplot as plt
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import RichProgressBar
from torch.utils.data import DataLoader
from torchvision import transforms as T

from sklearn.model_selection import train_test_split

import yaml
from pprint import pprint

with open('../conf/config_DV.yaml') as file:
    yaml_data = yaml.safe_load(file)

pprint(yaml_data)

#################### DEFINE CONSTANTS ####################

DATASET_ROOT_PATH = yaml_data['datasets']['retinopathy_dataset']['root_path']
TRAIN_DF_PATH = yaml_data['datasets']['retinopathy_dataset']['train_df_path']
TEST_DF_PATH = yaml_data['datasets']['retinopathy_dataset']['test_df_path']
VALIDATION_SPLIT = 0.3
SEED = 42
TRAIN_CAT_LABELS = yaml_data['datasets']['retinopathy_dataset']['train_cat_labels']
VAL_CAT_LABELS = yaml_data['datasets']['retinopathy_dataset']['val_cat_labels']
TEST_CAT_LABELS = yaml_data['datasets']['retinopathy_dataset']['test_cat_labels']
BATCH_SIZE = yaml_data['train']['batch_size']

train_transform = T.Compose([
    T.ToPILImage(),
    T.Resize((224,224)),
    T.ToTensor(),
    T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])


main_df = pd.read_csv(TRAIN_DF_PATH)
test_df = pd.read_csv(TEST_DF_PATH)

main_df['image'] = main_df['image'].apply(lambda x: str(DATASET_ROOT_PATH+'final_train/train/'+x))
test_df['image'] = test_df['image'].apply(lambda x: str(DATASET_ROOT_PATH+'final_test/test/'+x))

# Creating training and validation splits
train_df, val_df = train_test_split(main_df, test_size=VALIDATION_SPLIT,
                                    random_state=SEED)

train_df = train_df.reset_index(drop=True)
val_df = val_df.reset_index(drop=True)

print("Length of Main dataset: ", len(main_df))

#Creating Datasets
train_dataset = retinopathy_dataset.RetinopathyDataset(df=train_df, cat_labels_to_include=TRAIN_CAT_LABELS, 
                                                        transforms=train_transform)

val_dataset = retinopathy_dataset.RetinopathyDataset(df=val_df, cat_labels_to_include=VAL_CAT_LABELS, 
                                                        transforms=train_transform)

test_dataset = retinopathy_dataset.RetinopathyDataset(df=test_df, cat_labels_to_include=TEST_CAT_LABELS, 
                                                        transforms=train_transform)

print("Length of Training dataset: ", train_dataset.__len__())
print("Length of Validation dataset: ", val_dataset.__len__())
print("Length of Test dataset: ", test_dataset.__len__())

val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=12)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=12)

dm = retinopathy_dataset.LightningRetinopathyDataset(train_dataset, val_dataset,
                                                     test_dataset, BATCH_SIZE)





