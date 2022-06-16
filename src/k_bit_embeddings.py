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
from model import supervised_model

import argparse
import yaml
from pprint import pprint

##############################################################################################

parser = argparse.ArgumentParser(description='Hyper-parameters management')
parser.add_argument('--dataset', type=str, default='retinopathy', help='Dataset to use for training')
parser.add_argument('--experimental_run', type=bool, default=False, help='Experimental run (unit test)')
parser.add_argument('--num_runs', type=int, default=1, help='Number of runs for a dataset. Different augmentations will be randomly samples in each run.')
parser.add_argument('--train_mlp', type=bool, default=False, help='Train an MLP instead of a single layer')
parser.add_argument('--imagenet_policy', type=bool, default=False, help='Use ImageNet policy given by AutoAugment for augmentation.')

args = parser.parse_args()

with open('../conf/config_integrated.yaml') as file:
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
IMAGENET_POLICY = args.imagenet_policy

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

#All these transformations are defined in Albumentations
aug_dict = {
            CLAHE(): 1,
            ColorJitter(): 2,
            Downscale(): 3,
            Emboss(): 4,
            ShiftScaleRotate(): 5,
            HorizontalFlip(): 6,
            VerticalFlip(): 7,
            ImageCompression(): 8,
            Rotate(): 9,
            Normalize(): 10,
            Perspective(): 11
            }


all_test_acc = []
all_test_loss = []
all_test_f1 = []
all_val_acc = []
all_val_loss = []
all_val_f1 = []
all_aug_bits = []
all_runs = []

# A dictionary to store the best results and corresponding bit representation from all runs
all_results = {
                'test_acc': [],
                'test_loss': [],
                'test_f1': [],
                'val_acc': [],
                'val_loss': [],
                'val_f1': [],
                'k_bit_representation': [0]*len(aug_dict),
                'run': []
                }

for _run in range(NUM_RUNS):

    print(" ################## Starting Run {} ################## ".format(_run+1))
    all_runs.append(_run+1)

    #Load the train set
    main_df = pd.read_csv(TRAIN_DF_PATH)

    test_run_acc = 0         #Initialize new accuracy for each run
    test_run_loss = 0        #Initialize new loss for each run
    test_run_f1 = 0          #Initialize new F1 score for each run

    if(IMAGENET_POLICY):

        assert AUG_BIT_VECTOR == None

        EXPERIMENT = 'autoaugment_imagenet_policy_'
        policy = T.AutoAugmentPolicy.IMAGENET
        train_transform = T.Compose([T.Resize([224, 224]), T.AutoAugment(policy)])
        basic_transform = T.Compose([T.Resize([224, 224])])

    else:

        assert IMAGENET_POLICY == False

        if(AUG_BIT_VECTOR != None):

            assert len(AUG_BIT_VECTOR) == len(aug_dict)

            aug_bit = AUG_BIT_VECTOR
            _selected_augs = utils.get_aug_from_vector(aug_dict, AUG_BIT_VECTOR)

            if(np.sum(AUG_BIT_VECTOR) == len(aug_dict)):
                EXPERIMENT = 'bernoulli_k_bit_baseline_all_augs'
            elif(np.sum(AUG_BIT_VECTOR) == 0):
                EXPERIMENT = 'bernoulli_k_bit_baseline_no_augs'
            
        else:
            _selected_augs, aug_bit = utils.gen_binomial_dict(aug_dict)

        if(TRAIN_MLP):
            EXPERIMENT = EXPERIMENT + '_mlp_'

            

        print("Number of augmentations selected: {}".format(len(_selected_augs)))
        print("Augmentation bit representation: {}".format(aug_bit))
        all_aug_bits.append(aug_bit)

        #Add required basic transforms here.
        _selected_augs = [Resize(224, 224)] + _selected_augs

        train_transform = A.Compose(_selected_augs)
        basic_transform = A.Compose([Resize(224, 224)])


    ##################################### DATASETS & DATALOADERS ##########################

    results_dict = utils.get_dataloaders(yaml_data, DATASET, train_transform, basic_transform)

    train_image_loader = results_dict['train_image_loader']
    val_image_loader = results_dict['val_image_loader']
    test_image_loader = results_dict['test_image_loader']
    ACTIVATION = results_dict['activation']
    LOSS_FN = results_dict['loss_fn']
    MULTILABLE = results_dict['multilable']
    CLASS_WEIGHTS = results_dict['class_weights']

    #raise SystemExit(0)
    
    ######################################################################################


    model = supervised_model.SupervisedModel(encoder=ENCODER, batch_size = BATCH_SIZE, num_classes=NUM_CLASSES,
                                            class_weights = CLASS_WEIGHTS, lr_rate=lr_rate, lr_scheduler=LR_SCHEDULER, 
                                            do_finetune=DO_FINETUNE, train_mlp=TRAIN_MLP,
                                            activation=ACTIVATION, criterion=LOSS_FN, multilable=MULTILABLE)

    trainer = pl.Trainer(gpus=GPUs, 
                        max_epochs=EPOCHS,
                        )

    if(AUTO_LR_FIND):
        lr_finder = trainer.tuner.lr_find(model, train_image_loader, update_attr=True)
        new_lr = lr_finder.suggestion()
        print("New suggested learning rate is: ", new_lr)
        model.hparams.learning_rate = new_lr

    #In this case, we have train, validation, and test dataloaders for all datasets
    print("################ Initiating training process ####################")
    #trainer.fit(model, train_image_loader)
    trainer.fit(model, train_image_loader, val_image_loader)

    #print("################ Initiating validation process ##################")
    # val_results = trainer.validate(dataloaders=val_image_loader)

    # val_run_acc = val_results[0]['val_acc']
    # val_run_loss = val_results[0]['val_loss']
    # val_run_f1 = val_results[0]['f1_score']

    print("################ Initiating testing process #####################")
    test_results = trainer.test(dataloaders=test_image_loader)
    
    test_run_acc = test_results[0]['test_acc']
    test_run_loss = test_results[0]['test_loss']
    test_run_f1 = test_results[0]['f1_score']

    all_test_acc.append(test_run_acc)
    all_test_loss.append(test_run_loss)
    all_test_f1.append(test_run_f1)

    # all_val_acc.append(val_run_acc)
    # all_val_loss.append(val_run_loss)
    # all_val_f1.append(val_run_f1)
    #print(test_results)

    #Saving results from all the runs
    all_results['test_acc'] = all_test_acc
    all_results['test_loss'] = all_test_loss
    all_results['test_f1'] = all_test_f1
    # all_results['val_acc'] = all_val_acc
    # all_results['val_loss'] = all_val_loss
    # all_results['val_f1'] = all_val_f1
    all_results['k_bit_representation'] = all_aug_bits
    all_results['run'] = all_runs

    # print("Validation Accuracy for run {} is: {}".format(_run+1, val_run_acc))
    # print("Validation Loss for run {} is: {}".format(_run+1, val_run_loss))
    # print("Validation F1 Score for run {} is: {}".format(_run+1, val_run_f1))

    print("Test Accuracy for run {} is: {}".format(_run+1, test_run_acc))
    print("Test Loss for run {} is: {}".format(_run+1, test_run_loss))
    print("Test F1 Score for run {} is: {}".format(_run+1, test_run_f1))

    print("#######################################################")
    print('\n')

if(NUM_RUNS > 1):
    mean_test_acc = sum(all_test_acc)/len(all_test_acc)
    mean_test_loss = sum(all_test_loss)/len(all_test_loss)
    mean_test_f1 = sum(all_test_f1)/len(all_test_f1)

    # mean_val_acc = sum(all_val_acc)/len(all_val_acc)
    # mean_val_loss = sum(all_val_loss)/len(all_val_loss)
    # mean_val_f1 = sum(all_val_f1)/len(all_val_f1)

    # Standard Deviation in Test and Validation metrics

    print("Average Augmentation Bit Representation: ", [sum(i) for i in zip(*all_aug_bits)])
    print("Deviation from mean test accuracy in each run: ", [x - mean_test_acc for x in all_test_acc])
    print("Deviation from mean test loss in each run: ", [x - mean_test_loss for x in all_test_loss])
    print("Deviation from mean test F1 in each run: ", [x - mean_test_f1 for x in all_test_f1])
    print("\n")

    # print("Deviation from mean validation accuracy in each run: ", [x - mean_val_acc for x in all_val_acc])
    # print("Deviation from mean validation loss in each run: ", [x - mean_val_loss for x in all_val_loss])
    # print("Deviation from mean validation F1 in each run: ", [x - mean_val_f1 for x in all_val_f1])
    # print("\n")

    # # Standard Deviation in Test and Validation metrics
    # print("Standard Deviation for test accuracy across all runs: {}".format(np.std(all_test_acc)))
    # print("Standard Deviation for test loss across all runs: {}".format(np.std(all_test_loss)))
    # print("Standard Deviation for test F1 score across all runs: {}".format(np.std(all_test_f1)))
    # print("\n")

    # print("Standard Deviation for validation accuracy across all runs: {}".format(np.std(all_val_acc)))
    # print("Standard Deviation for validation loss across all runs: {}".format(np.std(all_val_loss)))
    # print("Standard Deviation for validation F1 score across all runs: {}".format(np.std(all_val_f1)))
    # print("\n")

    
    print("\n")

    info_dict = {'num_runs': NUM_RUNS,
                 'dataset': DATASET,
                 'encoder': ENCODER,
                 'finetune': DO_FINETUNE,
                 'experiment': EXPERIMENT}

    # utils.plot_run_stats(all_val_f1, all_test_f1, 
    #                     info_dict=info_dict,
    #                     save_dir='saved_plots/', 
    #                     save_plot=SAVE_PLOTS)

    
    #f.write("\n")
    if(LOGGING):
        f.write(" ########### SUMMARY ############ ")
        f.write("\n")
        f.write("Average Augmentation Bit Representation: {}".format([sum(i) for i in zip(*all_aug_bits)]))
        f.write("\n")
        # f.write("Deviation from mean accuracy in each run: {}".format([x - mean_test_acc for x in all_test_acc]))
        # f.write("\n")
        # f.write("Deviation from mean loss in each run: {}".format([x - mean_test_loss for x in all_test_loss]))
        # f.write("\n")
        # f.write("Deviation from mean F1 in each run:{} ".format([x - mean_test_f1 for x in all_test_f1]))
        # f.write("\n")
        # f.write("Standard Deviation for test accuracy across all runs: {}".format(np.std(all_test_acc)))
        # f.write("\n")
        # f.write("Standard Deviation for test loss across all runs: {}".format(np.std(all_test_loss)))
        # f.write("\n")
        # f.write("Standard Deviation for F1 score across all runs: {}".format(np.std(all_test_f1)))
        # f.write("\n")
        # f.write("\n")
        # f.write("\n")

        f.write(" ############# ALL RESULTS ############### \n")
        #print the all_results dictionary here after sorting
        sorted_results = utils.sort_dictionary(all_results)

        pprint(sorted_results, f)
        f.write("#############################################################################")
        f.write("\n")
        f.write("\n")
        f.write("\n")

        f.close()

