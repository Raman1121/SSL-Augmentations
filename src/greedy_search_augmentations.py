import os
import time
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
from albumentations.augmentations.geometric.resize import Resize

from dataset import retinopathy_dataset, cancer_mnist_dataset, mura_dataset, chexpert_dataset
from model import supervised_model

import argparse
import yaml
from pprint import pprint

##############################################################################################

parser = argparse.ArgumentParser(description='Hyper-parameters management')
parser.add_argument('--dataset', type=str, default='retinopathy', help='Dataset to use for training')
parser.add_argument('--experimental_run', type=bool, default=False, help='Experimental run (unit test)')

args = parser.parse_args()

start_time = time.time()

with open('../conf/config_greedy.yaml') as file:
        yaml_data = yaml.safe_load(file)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

if(device == torch.device(type='cpu')):
    GPUs = 0
elif(device == torch.device(type='cuda')):
    GPUs = 1
    #GPUs = torch.cuda.device_count()

#RUN CONSTANTS
BATCH_SIZE = yaml_data['run']['batch_size']
DATASET = args.dataset
lr_rate = yaml_data['run']['lr_rate']
TEST_EPOCHS = yaml_data['run']['test_epochs']
INNER_EPOCHS = yaml_data['run']['inner_epochs']
EXPERIMENTAL_RUN = args.experimental_run
EMBEDDINGS_DIM = 2048
SUBSET = yaml_data['run']['subset']
AUTO_LR_FIND = yaml_data['run']['auto_lr_find']
SAVE_PLOTS = yaml_data['run']['save_plots']
EXPERIMENT = yaml_data['run']['experiment']
LOG_FOLDER = yaml_data['run']['log_folder']
DO_FINETUNE = yaml_data['run']['do_finetune']
ENCODER = yaml_data['run']['encoder']
LR_SCHEDULER = yaml_data['run']['lr_scheduler']
LOGGING = True

#DATASET CONSTANTS
DATASET_ROOT_PATH = yaml_data['all_datasets'][DATASET]['root_path']
TRAIN_DF_PATH = yaml_data['all_datasets'][DATASET]['train_df_path']
VALIDATION_SPLIT = 0.3
TEST_SPLIT = 0.2
SEED = 42
NUM_CLASSES = yaml_data['all_datasets'][DATASET]['num_classes']


#SAVING CONSTANTS
SAVED_MODELS_DIR = '../Saved_models'

if(EXPERIMENTAL_RUN):
    INNER_EPOCHS = 1
    TEST_EPOCHS = 1
    AUTO_LR_FIND = False
    LOGGING = False

if(LOGGING):

    #Save results to a text file
    filename = EXPERIMENT + '_' + DATASET + '_' + ENCODER + '.txt'
    f = open(os.path.join(LOG_FOLDER, filename), "a")
    f.write("EXPERIMENT DATE: {}".format(date.today()))
    f.write("\n")

    pprint(yaml_data, f)
    print("Dataset: {}".format(DATASET), f)
    print("SUBSET: {}".format(SUBSET), f)
    print("Saved Models dir: {}".format(SAVED_MODELS_DIR), f)



transform_prob = 1
crop_height = 224
crop_width = 224

aug_dict = {CLAHE(p=transform_prob): 1,
            ColorJitter(p=transform_prob): 2,
            Downscale(p=transform_prob): 3,
            Emboss(p=transform_prob): 4,
            Flip(p=transform_prob): 5,
            HorizontalFlip(p=transform_prob): 6,
            VerticalFlip(p=transform_prob): 7,
            ImageCompression(p=transform_prob): 8,
            Rotate(p=transform_prob): 9
            }

# Make sure to maintain the same order of augmentations here as in aug_dict dictionary
aug_dict_labels = {'CLAHE': 1,
                   'ColorJitter': 2,
                   'Downscale': 3,
                   'Emboss': 4,
                   'Flip': 5,
                   'HorizontalFlip': 6,
                   'VerticalFlip': 7,
                   'ImageCompression': 8,
                   'Rotate': 9
                   }


all_aug_list = list(aug_dict.keys())
all_aug_labels_list = list(aug_dict_labels.keys())

all_selected_augs = []
all_selected_augs_labels = []

all_acc = []
all_loss = []
all_f1 = []
all_aug_bits = []
all_runs = []

# A dictionary to store the best augmentations and results from all runs
test_results_dict = {'aug':[],
                     'aug_label': [],
                     'f1':[]}

while(len(all_aug_list) > 0):

    # A dictionary to store the augmentations and their corresponding results for each pass
    _aug_results = {'aug':[],
                    'aug_label': [],
                    'f1':[]}

    for _aug, _aug_label in zip(all_aug_list, all_aug_labels_list):
        
        print("############# Initiating a new pass #############")

        
        run_acc = 0         #Initialize new accuracy for each pass
        run_loss = 0        #Initialize new loss for each pass
        run_f1 = 0          #Initialize new F1 score for each pass

        print("Selecting augmentation: {}".format(_aug_label))

        print("Number of augmentations remaining {}".format(len(all_aug_list)))

        #Add required basic transforms here.
        augs_for_this_pass = [Resize(224, 224)] + all_selected_augs + [_aug]
        

        train_transform = A.Compose(augs_for_this_pass)
        basic_transform = A.Compose([Resize(224, 224)])

        ##################################### DATASETS & DATALOADERS #######################################

        results_dict = utils.get_dataloaders(yaml_data, DATASET, train_transform, basic_transform)

        train_image_loader = results_dict['train_image_loader']
        val_image_loader = results_dict['val_image_loader']
        test_image_loader = results_dict['test_image_loader']
        ACTIVATION = results_dict['activation']
        LOSS_FN = results_dict['loss_fn']
        MULTILABLE = results_dict['multilable']
        CLASS_WEIGHTS = results_dict['class_weights']

        ######################################################################################

        model = supervised_model.SupervisedModel(encoder=ENCODER, batch_size = BATCH_SIZE, num_classes=NUM_CLASSES,
                                                class_weights = CLASS_WEIGHTS, lr_rate=lr_rate, lr_scheduler=LR_SCHEDULER, 
                                                do_finetune=DO_FINETUNE, 
                                                activation=ACTIVATION, criterion=LOSS_FN, multilable=MULTILABLE)

        trainer = pl.Trainer(gpus=GPUs, 
                            max_epochs=INNER_EPOCHS,
                            )

        if(AUTO_LR_FIND):
            lr_finder = trainer.tuner.lr_find(model, train_image_loader, update_attr=True)
            new_lr = lr_finder.suggestion()
            print("New suggested learning rate is: ", new_lr)
            model.hparams.learning_rate = new_lr

        #In this case, we have train, validation, and test dataloaders for all datasets
        trainer.fit(model, train_image_loader)

        #Perform validation here to obtain initial metrics
        validation_results = trainer.validate(dataloaders=val_image_loader)

        run_acc = validation_results[0]['val_acc']
        run_loss = validation_results[0]['val_loss']
        run_f1 = validation_results[0]['f1_score']
        
        #Add the current augmentation and f1 score to the results dictionary
        _aug_results['aug'] += [_aug]
        _aug_results['aug_label'] += [_aug_label]
        _aug_results['f1'] += [run_f1]

    #Obtain the augmentation which gave the best f1 score for this pass
    sorted_dict = {}
    augmentations = _aug_results['aug']
    augmentation_labels = _aug_results['aug_label']
    f1_scores = _aug_results['f1']

    #print("Augmentations: ", augmentations)
    print("f1 scores: ", f1_scores)

    l1, l2, l3 = (list(t) for t in zip(*sorted(zip(f1_scores, augmentations, augmentation_labels), reverse=True)))

    sorted_dict['f1'] = l1
    sorted_dict['aug'] = l2
    sorted_dict['aug_label'] = l3

    _best_augmentation = sorted_dict['aug'][0]
    _best_augmentation_label = sorted_dict['aug_label'][0]

    print("Best augmentation in this pass: {}".format(_best_augmentation))

    all_selected_augs.append(_best_augmentation)    #Add this augmentation to the list of selected augmentations
    all_selected_augs_labels.append(_best_augmentation_label)
    all_aug_list.remove(_best_augmentation)         #Remove this augmentation from the list of augmentations
    all_aug_labels_list.remove(_best_augmentation_label)

    # Add this selection to the test results dictionary

    test_results_dict['aug'] += [all_selected_augs[:]]
    test_results_dict['aug_label'] += [all_selected_augs_labels[:]]


print(test_results_dict)



print("############################################################# ")
print("Initializing Testing Process \n")

if(LOGGING):
    print("Initializing Testing Process \n", f)

greedy_augmentations_list = test_results_dict['aug']                #List of Lists
greedy_augmentations_labels_list = test_results_dict['aug_label']   #List of Lists

test_f1_scores = []

for augmentations_list, augmentations_labels_list in zip(greedy_augmentations_list, greedy_augmentations_labels_list):

    run_acc = 0
    run_loss = 0
    run_f1 = 0

    augs_for_this_pass = [Resize(224, 224)] + augmentations_list
    train_transform = A.Compose(augs_for_this_pass)
    basic_transform = A.Compose([Resize(224, 224)])

    ##################################### DATASETS & DATALOADERS #######################################

    results_dict = utils.get_dataloaders(yaml_data, DATASET, train_transform, basic_transform)

    train_image_loader = results_dict['train_image_loader']
    val_image_loader = results_dict['val_image_loader']
    test_image_loader = results_dict['test_image_loader']
    ACTIVATION = results_dict['activation']
    LOSS_FN = results_dict['loss_fn']
    MULTILABLE = results_dict['multilable']
    CLASS_WEIGHTS = results_dict['class_weights']

    #####################################################################################################

    model = supervised_model.SupervisedModel(encoder=ENCODER, batch_size = BATCH_SIZE, num_classes=NUM_CLASSES,
                                                class_weights = CLASS_WEIGHTS, lr_rate=lr_rate, lr_scheduler=LR_SCHEDULER, 
                                                do_finetune=DO_FINETUNE, 
                                                activation=ACTIVATION, criterion=LOSS_FN, multilable=MULTILABLE)

    trainer = pl.Trainer(gpus=GPUs, 
                        max_epochs=TEST_EPOCHS,
                        )

    if(AUTO_LR_FIND):
        lr_finder = trainer.tuner.lr_find(model, train_image_loader, update_attr=True)
        new_lr = lr_finder.suggestion()
        print("New suggested learning rate is: ", new_lr)
        model.hparams.learning_rate = new_lr

    #In this case, we have train, validation, and test dataloaders for all datasets
    trainer.fit(model, train_image_loader)

    test_results = trainer.test(dataloaders=test_image_loader)
        
    run_acc = test_results[0]['test_acc']
    run_loss = test_results[0]['test_loss']
    run_f1 = test_results[0]['f1_score']

    test_f1_scores.append(run_f1)

test_results_dict['f1'] = test_f1_scores

#Sort this dictionary
sorted_test_results_dict = {}
augmentations = test_results_dict['aug']
augmentations_labels = test_results_dict['aug_label']
f1_scores = test_results_dict['f1']

l1, l2, l3 = (list(t) for t in zip(*sorted(zip(f1_scores, augmentations, augmentations_labels), reverse=True)))

sorted_test_results_dict['f1'] = l1
sorted_test_results_dict['aug'] = l2
sorted_test_results_dict['aug_label'] = l3

print("############################ FINAL TEST RESULTS AFTER SORTING ############################")
print(sorted_test_results_dict)

if(LOGGING):
    pprint(sorted_test_results_dict, f)

print("################################### BEST AUGMENTATION AFTER GREEDY SEARCH ##########################")
print(sorted_test_results_dict['aug'][0])

if(LOGGING):
    pprint("Best augmentation after greedy search: {}".format(sorted_test_results_dict['aug'][0]), f)

print("################################### BEST F1 SCORE AFTER GREEDY SEARCH ##########################")
print(sorted_test_results_dict['f1'][0])

if(LOGGING):
    pprint("Best F1 Score after greedy search: {}".format(sorted_test_results_dict['f1'][0]), f)

info_dict = {
                 'dataset': DATASET,
                 'encoder': ENCODER,
                 'finetune': DO_FINETUNE,
                 'experiment': EXPERIMENT
            }

aug_bit_vector = utils.plot_greedy_augmentations(aug_dict, aug_dict_labels, 
                                sorted_test_results_dict,
                                info_dict,
                                save_plot=SAVE_PLOTS)

end_time = time.time()

print("Augmentation Bit Representation: {}".format(aug_bit_vector))
print("Execution Time: {}".format(end_time - start_time))


if(LOGGING):
    pprint("Augmentation Bit Representation: {}".format(aug_bit_vector), f)
    pprint("Execution Time: {}".format(end_time - start_time), f)



#raise SystemExit(0)













