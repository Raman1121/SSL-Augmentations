import os
import numpy as np
import pandas as pd
import pytest
import yaml
import random
from pprint import pprint

from utils import utils

import torchvision.transforms as T
import albumentations as A
from albumentations.augmentations.transforms import *
from albumentations.augmentations.crops.transforms import *
from albumentations.augmentations.geometric.rotate import *
from albumentations.augmentations.geometric.transforms import *
from albumentations.augmentations.geometric.resize import Resize


def load_config_file(filepath):
    with open(filepath) as file:
        yaml_data = yaml.safe_load(file)

    return yaml_data

def create_torchvision_transform():

    train_transform = T.Compose([T.Resize([224, 224]), T.AutoAugment(T.AutoAugmentPolicy.IMAGENET)])
    basic_transform = T.Compose([T.Resize([224, 224])])

    return train_transform, basic_transform

def create_albumentations_transform():

    _list = [CLAHE(), ColorJitter(), Downscale(), Emboss(), ShiftScaleRotate(), HorizontalFlip()]

    _selected_augs = [Resize(224, 224)] + random.sample(_list, 5)
    train_transform = A.Compose(_selected_augs)
    basic_transform = A.Compose([Resize(224, 224)])

    return train_transform, basic_transform



def load_dataloaders(dataset_name, config_file, transform_type):
    conf_folder = '../../conf/'
    yaml_data = load_config_file(os.path.join(conf_folder, config_file))

    assert transform_type in ['torchvision', 'albumentations']

    if(transform_type == 'torchvision'):
        train_transform, basic_transform = create_torchvision_transform()

    elif(transform_type == 'albumentations'):
        train_transform, basic_transform = create_albumentations_transform()

    results_dict = utils.get_dataloaders(yaml_data, dataset_name, train_transform, basic_transform)

    return results_dict
        
    
