import sys
sys.path.append('/home/co-dutt1/rds/hpc-work/SSL-Augmentations/src')

from utils import utils
import pytest
import os
from pprint import pprint
import yaml

from tests.conftest import *



@pytest.mark.parametrize(
    "filename",
    [
        "config_greedy.yaml",
        "config_integrated.yaml",
    ]
)
def test_load_config_file(filename, verbose=False):

    conf_folder = '../../conf/'
    yaml_data = load_config_file(os.path.join(conf_folder, filename))

    if(verbose):
        pprint(yaml_data)


@pytest.mark.parametrize(
    "dataset_name",
    [
        "retinopathy",
        "chexpert",
        "cancer_mnist",
        "mura"
    ]

)

def test_dataloaders_with_augs(dataset_name):

    conf_folder = '../../conf/'
    config_files = ["config_greedy.yaml", "config_integrated.yaml"]
    transform_types = ['torchvision', 'albumentations']

    for _conf_file in config_files:

        for _transform_type in transform_types:
    
            results_dict = load_dataloaders(dataset_name, _conf_file, _transform_type)

            assert results_dict != None

            train_image_loader = results_dict['train_image_loader']
            val_image_loader = results_dict['val_image_loader']
            test_image_loader = results_dict['test_image_loader']
            ACTIVATION = results_dict['activation']
            LOSS_FN = results_dict['loss_fn']
            MULTILABLE = results_dict['multilable']
            CLASS_WEIGHTS = results_dict['class_weights']

            assert train_image_loader != None
            assert val_image_loader != None
            assert test_image_loader != None
            assert ACTIVATION != None
            assert LOSS_FN != None
            assert MULTILABLE != None

            if(dataset_name == 'chexpert'):
                continue
            else:
                assert CLASS_WEIGHTS != None
    


