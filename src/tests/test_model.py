import sys
sys.path.append('/home/co-dutt1/rds/hpc-work/SSL-Augmentations/src')

from utils import utils
from model import *
import pytest
import os
from pprint import pprint
import yaml

from model import supervised_model

from tests.conftest import *


@pytest.mark.parametrize(
    "dataset_name",
    [
        "retinopathy",
        "chexpert",
        "cancer_mnist",
        "mura"
    ]

)
def test_load_supervised_model(dataset_name):
    conf_folder = '../../conf/'
    config_files = ["config_greedy.yaml", "config_integrated.yaml"]

    for _conf_file in config_files:
        yaml_data = load_config_file(os.path.join(conf_folder, _conf_file))

        ENCODER = yaml_data['run']['encoder']
        BATCH_SIZE = yaml_data['run']['batch_size']
        NUM_CLASSES = yaml_data['all_datasets'][dataset_name]['num_classes']
        lr_rate = yaml_data['run']['lr_rate']
        LR_SCHEDULER = yaml_data['run']['lr_scheduler']
        DO_FINETUNE = yaml_data['run']['do_finetune']
        TRAIN_MLP = True
        transform_type = 'albumentations'

        results_dict = load_dataloaders(dataset_name, _conf_file, transform_type)

        ACTIVATION = results_dict['activation']
        LOSS_FN = results_dict['loss_fn']
        MULTILABLE = results_dict['multilable']
        CLASS_WEIGHTS = results_dict['class_weights']

        model = supervised_model.SupervisedModel(encoder=ENCODER, batch_size = BATCH_SIZE, num_classes=NUM_CLASSES,
                                            class_weights = CLASS_WEIGHTS, lr_rate=lr_rate, lr_scheduler=LR_SCHEDULER, 
                                            do_finetune=DO_FINETUNE, train_mlp=TRAIN_MLP,
                                            activation=ACTIVATION, criterion=LOSS_FN, multilable=MULTILABLE)

        assert model != None





