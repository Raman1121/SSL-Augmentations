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

if(LOGGING):

    #Save results to a text file
    filename = EXPERIMENT + '_' + DATASET + '_' + ENCODER + '_' + str(NUM_RUNS) + '.txt'
    f = open(os.path.join(LOG_FOLDER, filename), "a")
    f.write("EXPERIMENT DATE: {}".format(date.today()))
    f.write("\n")

    pprint(yaml_data, f)
    print("Dataset: {}".format(DATASET), f)
    print("SUBSET: {}".format(SUBSET), f)
    print("Number of Runs: {}".format(NUM_RUNS), f)
    print("Saved Models dir: {}".format(SAVED_MODELS_DIR), f)



# logging.basicConfig(filename=EXPERIMENT+'_'+DATASET+'.log')
# logging.info("YAML DATA: f'{yaml_data}")

transform_prob = 1
#crop_height = int(0.7*224) #Assuming all images would be 224x224
#crop_width = int(0.7*224)  #Assuming all images would be 224x224

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
            Rotate(p=transform_prob): 9}


all_acc = []
all_loss = []
all_f1 = []
all_aug_bits = []
all_runs = []

# A dictionary to store the best results and corresponding bit representation from all runs
all_results = {
                'acc': [],
                'loss': [],
                'f1': [],
                'k_bit_representation': [0]*len(aug_dict),
                'run': []
                }

for _run in range(NUM_RUNS):

    print(" ################## Starting Run {} ################## ".format(_run+1))
    all_runs.append(_run+1)

    #Load the train set
    main_df = pd.read_csv(TRAIN_DF_PATH)

    run_acc = 0         #Initialize new accuracy for each run
    run_loss = 0        #Initialize new loss for each run
    run_f1 = 0          #Initialize new F1 score for each run

    # aug_bit = [0]*len(aug_dict)

    # # print("Number of augmentations: ", len(aug_dict))
    # # randomly_selected_augs = random.sample(aug_list, int(0.7*len(aug_list)))
    # randomly_selected_augs = random.sample(list(aug_dict), int(0.7*len(aug_dict)))

    # print("Number of augmentations selected: {}".format(len(randomly_selected_augs)))
    # #print("Augmentations selected: {} \n".format([i for i in randomly_selected_augs]))

    # for aug in randomly_selected_augs:
    #     index = aug_dict[aug] - 1
    #     aug_bit[index] = 1
        
    # print(aug_bit)
    # all_aug_bits.append(aug_bit)

    randomly_selected_augs, aug_bit = utils.gen_binomial_dict(aug_dict)
    print("Number of augmentations selected: {}".format(len(randomly_selected_augs)))
    #print("Augmentations selected: {} \n".format([i for i in randomly_selected_augs]))
    print("Augmentation bit representation: {}".format(aug_bit))
    all_aug_bits.append(aug_bit)

    # raise SystemExit(0)

    #Add required basic transforms here.
    randomly_selected_augs = [Resize(224, 224)] + randomly_selected_augs

    train_transform = A.Compose(randomly_selected_augs)
    basic_transform = A.Compose([Resize(224, 224)])

    ##################################### DATASETS #######################################

    train_dataset = None
    test_dataset = None
    val_dataset = None

    if(DATASET == 'retinopathy'):

        '''
        Preparing the Diabetic Retinopathy dataset
        '''

        CLASS_WEIGHTS = compute_class_weight(class_weight = 'balanced', 
                                             classes = np.unique(main_df['level']), 
                                             y = main_df['level'].to_numpy())

        CLASS_WEIGHTS = torch.Tensor(CLASS_WEIGHTS)
        
        #Load the test set for DR dataset
        TEST_DF_PATH = yaml_data['all_datasets'][DATASET]['test_df_path']
        test_df = pd.read_csv(TEST_DF_PATH)

        TRAIN_CAT_LABELS = yaml_data['all_datasets'][DATASET]['train_cat_labels']
        VAL_CAT_LABELS = yaml_data['all_datasets'][DATASET]['val_cat_labels']
        TEST_CAT_LABELS = yaml_data['all_datasets'][DATASET]['test_cat_labels']

        main_df['image'] = main_df['image'].apply(lambda x: str(DATASET_ROOT_PATH+'final_train/train/'+x))
        test_df['image'] = test_df['image'].apply(lambda x: str(DATASET_ROOT_PATH+'final_test/test/'+x))

        #Checking if SUBSET size is greater than the size of the dataset itself.
        TRAIN_SUBSET = len(main_df) if len(main_df) < int(0 if SUBSET == None else SUBSET) else SUBSET
        TEST_SUBSET = len(test_df) if len(test_df) < int(0 if SUBSET == None else SUBSET) else SUBSET

        train_dataset = retinopathy_dataset.RetinopathyDataset(df=main_df, cat_labels_to_include=TRAIN_CAT_LABELS, 
                                                            transforms=train_transform, subset=TRAIN_SUBSET)

        test_dataset = retinopathy_dataset.RetinopathyDataset(df=test_df, cat_labels_to_include=TEST_CAT_LABELS, 
                                                            transforms=basic_transform, subset=TEST_SUBSET)
                                                        
        ACTIVATION = 'softmax'
        LOSS_FN = 'cross_entropy'
        MULTILABLE = False

    elif(DATASET == 'cancer_mnist'):

        '''
        Preparing the Cancer MNIST dataset
        '''

        #NOTE: Val and Test data for this dataset has not been provided!

        CLASS_WEIGHTS = compute_class_weight(class_weight = 'balanced', 
                                             classes = np.unique(main_df['cell_type_idx']), 
                                             y = main_df['cell_type_idx'].to_numpy())

        CLASS_WEIGHTS = torch.Tensor(CLASS_WEIGHTS)

        # Creating training and test splits
        train_df, test_df = train_test_split(main_df, test_size=VALIDATION_SPLIT,
                                    random_state=SEED)
        train_df = train_df.reset_index(drop=True)
        test_df = test_df.reset_index(drop=True)

        #Checking if SUBSET size is greater than the size of the dataset itself.

        TRAIN_SUBSET = len(train_df) if len(train_df) < int(0 if SUBSET == None else SUBSET) else SUBSET
        TEST_SUBSET = len(test_df) if len(test_df) < int(0 if SUBSET == None else SUBSET) else SUBSET

        train_dataset = cancer_mnist_dataset.CancerMNISTDataset(df=train_df, transforms=train_transform, 
                                                                subset=TRAIN_SUBSET)

        test_dataset = cancer_mnist_dataset.CancerMNISTDataset(df=test_df, transforms=basic_transform, 
                                                                subset=TEST_SUBSET)

        ACTIVATION = 'softmax'
        LOSS_FN = 'cross_entropy'
        MULTILABLE = False

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
        
        #Checking if SUBSET size is greater than the size of the dataset itself.

        TRAIN_SUBSET = len(train_df) if len(train_df) < int(0 if SUBSET == None else SUBSET) else SUBSET
        VAL_SUBSET = len(val_df) if len(val_df) < int(0 if SUBSET == None else SUBSET) else SUBSET
        TEST_SUBSET = len(test_df) if len(test_df) < int(0 if SUBSET == None else SUBSET) else SUBSET
    
        train_dataset = chexpert_dataset.ChexpertDataset(df=train_df, transforms=train_transform, 
                                                        subset=TRAIN_SUBSET)

        val_dataset = chexpert_dataset.ChexpertDataset(df=val_df, transforms=train_transform,
                                                        subset=VAL_SUBSET)

        test_dataset = chexpert_dataset.ChexpertDataset(df=test_df, transforms=basic_transform, 
                                                        subset=TEST_SUBSET)
                                                

        ACTIVATION = 'sigmoid'
        LOSS_FN = 'bce'
        MULTILABLE = True
        CLASS_WEIGHTS = None

    elif(DATASET == 'mura'):
        '''
        Preparing the MURA dataset
        '''

        #NOTE: Test data for this dataset has not been provided!

        CLASS_WEIGHTS = compute_class_weight(class_weight = 'balanced', 
                                             classes = np.unique(main_df['label']), 
                                             y = main_df['label'].to_numpy())

        CLASS_WEIGHTS = torch.Tensor(CLASS_WEIGHTS)

        VAL_DF_PATH = yaml_data['all_datasets'][DATASET]['val_df_path']
        val_df = pd.read_csv(VAL_DF_PATH)

        # Creating training and test splits
        train_df, test_df = train_test_split(main_df, test_size=VALIDATION_SPLIT,
                                    random_state=SEED)

        train_df = train_df.reset_index(drop=True)
        test_df = test_df.reset_index(drop=True)

        #Checking if SUBSET size is greater than the size of the dataset itself.

        TRAIN_SUBSET = len(train_df) if len(train_df) < int(0 if SUBSET == None else SUBSET) else SUBSET
        VAL_SUBSET = len(val_df) if len(val_df) < int(0 if SUBSET == None else SUBSET) else SUBSET
        TEST_SUBSET = len(test_df) if len(test_df) < int(0 if SUBSET == None else SUBSET) else SUBSET

        train_dataset = mura_dataset.MuraDataset(df=train_df, transforms=train_transform, 
                                                                subset=TRAIN_SUBSET)

        val_dataset = mura_dataset.MuraDataset(df=val_df, transforms=train_transform,
                                                                subset=VAL_SUBSET)

        test_dataset = mura_dataset.MuraDataset(df=test_df, transforms=basic_transform, 
                                                                subset=TEST_SUBSET)

        ACTIVATION = 'softmax'
        LOSS_FN = 'cross_entropy'
        MULTILABLE = False

    ######################################################################################


    #Creating Data Loaders
    if(train_dataset != None):
        print("Train dataset length: ", len(train_dataset))
        train_image_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=12)

    if(val_dataset != None):
        print("Val dataset length: ", len(val_dataset))
        val_image_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=12)

    if(test_dataset != None):
        print("Test dataset length: ", len(test_dataset))
        test_image_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=12)

    
    model = supervised_model.SupervisedModel(encoder=ENCODER, batch_size = BATCH_SIZE, num_classes=NUM_CLASSES,
                                            class_weights = CLASS_WEIGHTS, lr_rate=lr_rate, lr_scheduler=LR_SCHEDULER, 
                                            do_finetune=DO_FINETUNE, 
                                            activation=ACTIVATION, criterion=LOSS_FN, multilable=MULTILABLE)

    trainer = pl.Trainer(gpus=GPUs, 
                        max_epochs=EPOCHS,
                        )

    if(AUTO_LR_FIND):
        lr_finder = trainer.tuner.lr_find(model, train_image_loader, update_attr=True)
        new_lr = lr_finder.suggestion()
        print("New suggested learning rate is: ", new_lr)
        model.hparams.learning_rate = new_lr

    if(val_dataset == None):

        print("Validation dataset not provided for {} dataset".format(DATASET))
        #Providing data loader for only the train set in the fit method.
        trainer.fit(model, train_image_loader)

    elif(val_dataset != None):

        #Providing data loader for both the train and val set in the fit method.
        trainer.fit(model, train_image_loader, val_image_loader)

    if(test_dataset != None):
        test_results = trainer.test(dataloaders=test_image_loader)
        
        run_acc = test_results[0]['test_acc']
        run_loss = test_results[0]['test_loss']
        run_f1 = test_results[0]['f1_score']

        # try:

        #     if(run_acc > best_results['best_acc'][0]):
        #         best_results['run'] = [_run+1]
        #         best_results['best_acc'] = [run_acc]
        #         best_results['k_bit_representation'] = [aug_bit]

        #     elif(run_acc == best_results['best_acc']):
        #         _best_runs = best_results['run']
        #         _best_runs.append(_run)

        #         _best_acc = best_results['best_acc']
        #         _best_acc.append(run_acc)

        #         _best_k_bits = best_results['k_bit_representation']
        #         _best_k_bits.append(aug_bit)

        #         best_results['run'] = _best_runs
        #         best_results['best_acc'] = _best_acc
        #         best_results['k_bit_representation'] = _best_k_bits
        
        # except:
        #     print("Error while obtaining best results from all runs.")

        all_acc.append(run_acc)
        all_loss.append(run_loss)
        all_f1.append(run_f1)
        #print(test_results)

        #Saving results from all the runs
        all_results['acc'] = all_acc
        all_results['loss'] = all_loss
        all_results['f1'] = all_f1
        all_results['k_bit_representation'] = all_aug_bits
        all_results['run'] = all_runs

        print("Test Accuracy for run {} is: {}".format(_run+1, run_acc))
        print("Test Loss for run {} is: {}".format(_run+1, run_loss))
        print("F1 Score for run {} is: {}".format(_run+1, run_f1))
        print("#######################################################")
        print('\n')



    else:
        print("Test data not provided for {} dataset hence, skipping testing.".format(DATASET))

if(NUM_RUNS > 1):
    mean_test_acc = sum(all_acc)/len(all_acc)
    mean_test_loss = sum(all_loss)/len(all_loss)
    mean_test_f1 = sum(all_f1)/len(all_f1)

    print("Average Augmentation Bit Representation: ", [sum(i) for i in zip(*all_aug_bits)])
    print("Deviation from mean accuracy in each run: ", [x - mean_test_acc for x in all_acc])
    print("Deviation from mean loss in each run: ", [x - mean_test_loss for x in all_loss])
    print("Deviation from mean F1 in each run: ", [x - mean_test_f1 for x in all_f1])
    print("\n")

    print("Standard Deviation for test accuracy across all runs: {}".format(np.std(all_acc)))
    print("Standard Deviation for test loss across all runs: {}".format(np.std(all_loss)))
    print("Standard Deviation for F1 score across all runs: {}".format(np.std(all_f1)))
    print("\n")

    
    print("\n")

    info_dict = {'num_runs': NUM_RUNS,
                 'dataset': DATASET,
                 'encoder': ENCODER,
                 'finetune': DO_FINETUNE,
                 'experiment': EXPERIMENT}

    utils.plot_run_stats(all_acc, all_loss, 
                        info_dict=info_dict,
                        save_dir='saved_plots/', 
                        save_plot=SAVE_PLOTS)

    
    #f.write("\n")
    if(LOGGING):
        f.write(" ########### SUMMARY ############ ")
        f.write("\n")
        f.write("Average Augmentation Bit Representation: {}".format([sum(i) for i in zip(*all_aug_bits)]))
        f.write("\n")
        f.write("Deviation from mean accuracy in each run: {}".format([x - mean_test_acc for x in all_acc]))
        f.write("\n")
        f.write("Deviation from mean loss in each run: {}".format([x - mean_test_loss for x in all_loss]))
        f.write("\n")
        f.write("Deviation from mean F1 in each run:{} ".format([x - mean_test_f1 for x in all_f1]))
        f.write("\n")
        f.write("Standard Deviation for test accuracy across all runs: {}".format(np.std(all_acc)))
        f.write("\n")
        f.write("Standard Deviation for test loss across all runs: {}".format(np.std(all_loss)))
        f.write("\n")
        f.write("Standard Deviation for F1 score across all runs: {}".format(np.std(all_f1)))
        f.write("\n")
        f.write("\n")
        f.write("\n")

        f.write(" ############# ALL RESULTS ############### \n")
        #print the all_results dictionary here after sorting
        sorted_results = utils.sort_dictionary(all_results)

        pprint(sorted_results, f)
        f.write("#############################################################################")
        f.write("\n")
        f.write("\n")
        f.write("\n")

        f.close()

