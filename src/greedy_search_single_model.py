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
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import ModelCheckpoint

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
from augs import standard_augmentations

import argparse
import yaml
from pprint import pprint

##############################################################################################

parser = argparse.ArgumentParser(description='Hyper-parameters management')
parser.add_argument('--dataset', type=str, default='retinopathy', help='Dataset to use for training')
parser.add_argument('--experimental_run', type=bool, default=False, help='Experimental run (unit test)')
parser.add_argument('--train_mlp', type=bool, default=False, help='Train an MLP instead of a single layer')
parser.add_argument('--use_mean_embeddings', type=bool, default=False, help='Use mean embeddings for running the experiments')

parser.add_argument('--encoder', type=str, default='resnet50', help='Encoder to be used for the experiments')
parser.add_argument('--do_finetune', action='store_true', help='Whether to do fine-tuning or not.')
parser.add_argument('--auto_lr_find', action='store_true', help='Whether to find learning rate automatically.')
parser.add_argument('--lr', type=float, default=0.0003, help='Learning Rate')
parser.add_argument('--pretrained', type=bool, default=True, help='Whether to use a pretrained encoder or not.')
parser.add_argument('--metric', type=str, default='accuracy', help='Metric to be used for augmentation selection.')

args = parser.parse_args()

start_time = time.time()

with open('../conf/config_greedy.yaml') as file:
        yaml_data = yaml.safe_load(file)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

if(device == torch.device(type='cpu')):
    GPUs = 0
elif(device == torch.device(type='cuda')):
    #GPUs = 1
    GPUs = torch.cuda.device_count()

#RUN CONSTANTS
BATCH_SIZE = yaml_data['run']['batch_size']
TEST_EPOCHS = yaml_data['run']['test_epochs']
INNER_EPOCHS = yaml_data['run']['inner_epochs']
EMBEDDINGS_DIM = 2048
SUBSET = yaml_data['run']['subset']
SAVE_PLOTS = yaml_data['run']['save_plots']
EXPERIMENT = yaml_data['run']['experiment']
LOG_FOLDER = yaml_data['run']['log_folder']
LR_SCHEDULER = yaml_data['run']['lr_scheduler']
LOGGING = True

# CLI ARGUEMENTS
EXPERIMENTAL_RUN = args.experimental_run
lr_rate = args.lr
AUTO_LR_FIND = args.auto_lr_find
PRETRAINED = args.pretrained
TRAIN_MLP = args.train_mlp
USE_MEAN_EMBEDDINGS = args.use_mean_embeddings
DO_FINETUNE = args.do_finetune
ENCODER = args.encoder
DATASET = args.dataset
METRIC = args.metric

#DATASET CONSTANTS
DATASET_ROOT_PATH = yaml_data['all_datasets'][DATASET]['root_path']
TRAIN_DF_PATH = yaml_data['all_datasets'][DATASET]['train_df_path']
VALIDATION_SPLIT = 0.3
TEST_SPLIT = 0.2
SEED = 42
NUM_CLASSES = yaml_data['all_datasets'][DATASET]['num_classes']

all_identifiers = [int(x) for x in os.listdir('greedy_search_models')]
IDENTIFIER = random.randint(0, 1000)

while(IDENTIFIER in all_identifiers):
    IDENTIFIER = random.randint(0, 1000)

if(METRIC == 'accuracy'):
    EXPERIMENT = EXPERIMENT + '_with_acc'
else:
    EXPERIMENT = EXPERIMENT + '_with_F1'

# raise SystemExit(0)


#SAVING CONSTANTS
SAVED_MODELS_DIR = '../Saved_models'

if(EXPERIMENTAL_RUN):
    INNER_EPOCHS = 3
    TEST_EPOCHS = 1
    AUTO_LR_FIND = False
    LOGGING = False

if(LOGGING):

    if(DO_FINETUNE == True):
        ft_label = 'with_FT'
    elif(DO_FINETUNE == False):
        ft_label = 'without_FT'

    if(TRAIN_MLP == True):
        mlp_label = 'with_MLP'
    elif(TRAIN_MLP == False):
        mlp_label = 'without_MLP'

    if(USE_MEAN_EMBEDDINGS == True):
        mean_embeddings_label = 'with_ME'
    elif(USE_MEAN_EMBEDDINGS == False):
        mean_embeddings_label = 'without_ME'

    if(AUTO_LR_FIND == True):
        auto_lr_label = 'with_auto_LR'
    elif(AUTO_LR_FIND == False):
        auto_lr_label = 'without_auto_LR'

    #Save results to a text file
    filename = EXPERIMENT + '_' + DATASET + '_' + ENCODER + '_' + ft_label + '_' + mlp_label + '_' + mean_embeddings_label + '_' + auto_lr_label + '.txt'
    f = open(os.path.join(LOG_FOLDER, filename), "w")

    pprint(yaml_data, f)
    print("Dataset: {}".format(DATASET), f)
    print("SUBSET: {}".format(SUBSET), f)
    print("Saved Models dir: {}".format(SAVED_MODELS_DIR), f)

crop_height = 224
crop_width = 224

pprint(yaml_data)

#Importing augmentations
augs = standard_augmentations.StandardAugmentations(shuffle=True)   #Augmentations would be obtained in a shuffled order each time
new_aug_dict = augs.new_aug_dict

all_aug_list = list(new_aug_dict.values())
all_aug_labels_list = list(new_aug_dict.keys())

#all_aug_list.insert(0, None)
#all_aug_labels_list.insert(0, 'No Aug')

all_selected_augs = []
all_selected_augs_labels = []

all_acc = []
all_loss = []
all_f1 = []
all_aug_bits = []
all_runs = []

run_acc_no_aug = 0
run_loss_no_aug = 0
run_f1_no_aug = 0

test_results_dict = {'aug':[],
                     'aug_label': [],
                     'metric':[]
                     }

val_results_dict = { 'aug':[],
                     'aug_label': [],
                     'metric':[]
                     }

pprint(yaml_data)

#######################################################################################

'''
Running the loop for no augmentations case first
'''
print("Running the loop for no augmentations case first")

_aug_results = {
                    'aug':[],
                    'aug_label': [],
                    'metric':[]
                }

run_acc = 0         
run_loss = 0        
run_f1 = 0          

train_transform = A.Compose([Resize(224, 224)])
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


if(USE_MEAN_EMBEDDINGS):
            
    model = mean_embedding_model.MeanEmbeddingModel(encoder=ENCODER, batch_size = BATCH_SIZE, num_classes=NUM_CLASSES,
                                        class_weights = CLASS_WEIGHTS, lr_rate=lr_rate, lr_scheduler=LR_SCHEDULER, 
                                        do_finetune=DO_FINETUNE, train_mlp=TRAIN_MLP,
                                        activation=ACTIVATION, criterion=LOSS_FN, multilable=MULTILABLE,
                                        aug_list = aug_labels_for_this_pass, new_aug_dict=new_aug_dict, k=5, pretrained=PRETRAINED)

else:

    model = supervised_model.SupervisedModel(encoder=ENCODER, batch_size = BATCH_SIZE, num_classes=NUM_CLASSES,
                                            class_weights = CLASS_WEIGHTS, lr_rate=lr_rate, lr_scheduler=LR_SCHEDULER, 
                                            do_finetune=DO_FINETUNE, train_mlp=TRAIN_MLP,
                                            activation=ACTIVATION, criterion=LOSS_FN, multilable=MULTILABLE)


es = EarlyStopping('train_acc', check_finite=True, patience=20, verbose=True, mode="max")

trainer = pl.Trainer(gpus=GPUs, 
                        max_epochs=INNER_EPOCHS,
                        callbacks=[es],
                        log_every_n_steps=22
                    )

if(AUTO_LR_FIND):
    lr_finder = trainer.tuner.lr_find(model, train_image_loader, update_attr=True)
    new_lr = lr_finder.suggestion()
    print("New suggested learning rate is: ", new_lr)
    model.hparams.learning_rate = new_lr

# Train on the train set
trainer.fit(model, train_image_loader)

# Obtain metrics on the validation set        
validation_results = trainer.validate(dataloaders=val_image_loader)

run_acc = validation_results[0]['val_acc']
run_loss = validation_results[0]['val_loss']
run_f1 = validation_results[0]['f1_score']

if(METRIC == 'accuracy'):
    _aug_results['metric'] += [run_acc]
else:
    _aug_results['metric'] += [run_f1]

val_results_dict['aug'] += ['No Augmentation']
val_results_dict['aug_label'] += ['No Aug']
val_results_dict['metric'] += [_aug_results['metric'][0]]

print("################ Testing with no augmentations #####################")

test_results = trainer.test(dataloaders=test_image_loader)

test_acc = test_results[0]['test_acc']
test_loss = test_results[0]['test_loss']
test_f1 = test_results[0]['f1_score']

test_results_dict['aug'] += ['No Augmentation']
test_results_dict['aug_label'] += ['No Aug']

if(METRIC == 'accuracy'):
    test_results_dict['metric'] += [test_acc]
else:
    test_results_dict['metric'] += [test_f1]

#######################################################################################

print(" ############## Initializing Greedy Search ############## ")
_pass = 0
while(len(all_aug_list) > 0):
    print("Number of augmentations remaining {}".format(len(all_aug_list)))

    _pass += 1

    _aug_results = {
                    'aug':[],
                    'aug_label': [],
                    'metric':[]
                    }

    _model_paths_dict = {}
                        
                        

    print('\n')
    print("############# Initiating a new pass #############")

    for _aug, _aug_label in zip(all_aug_list, all_aug_labels_list):

        run_acc = 0         #Initialize new accuracy for each pass
        run_loss = 0        #Initialize new loss for each pass
        run_f1 = 0          #Initialize new F1 score for each pass


        # if(_pass == 1):
        #     #No Augmentations case
        #     print("No augmentations case")
        #     augs_for_this_pass = [Resize(224, 224)] + all_selected_augs + []
        #     aug_labels_for_this_pass = all_selected_augs_labels + ['No Aug']

            
        
        #Add required basic transforms here.
        print("Selecting augmentation: {}".format(_aug_label))
        augs_for_this_pass = [Resize(224, 224)] + all_selected_augs + [_aug]
        aug_labels_for_this_pass = all_selected_augs_labels + [_aug_label]

        print("#####################################")
        print("augs for this pass: ")
        print(all_selected_augs_labels, '+', [_aug_label])

        if(USE_MEAN_EMBEDDINGS):
            EXPERIMENT = 'mean_embeddings_'
            train_transform = None
            basic_transform = None
        
        else:
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

        #raise SystemExit(0)

        ######################################################################################

        if(USE_MEAN_EMBEDDINGS):
            
            model = mean_embedding_model.MeanEmbeddingModel(encoder=ENCODER, batch_size = BATCH_SIZE, num_classes=NUM_CLASSES,
                                                class_weights = CLASS_WEIGHTS, lr_rate=lr_rate, lr_scheduler=LR_SCHEDULER, 
                                                do_finetune=DO_FINETUNE, train_mlp=TRAIN_MLP,
                                                activation=ACTIVATION, criterion=LOSS_FN, multilable=MULTILABLE,
                                                aug_list = aug_labels_for_this_pass, new_aug_dict=new_aug_dict, k=5, pretrained=PRETRAINED)

        else:

            model = supervised_model.SupervisedModel(encoder=ENCODER, batch_size = BATCH_SIZE, num_classes=NUM_CLASSES,
                                                    class_weights = CLASS_WEIGHTS, lr_rate=lr_rate, lr_scheduler=LR_SCHEDULER, 
                                                    do_finetune=DO_FINETUNE, train_mlp=TRAIN_MLP,
                                                    activation=ACTIVATION, criterion=LOSS_FN, multilable=MULTILABLE)

        #Defining Callbacks

        model_dirpath = os.path.join('greedy_search_models', str(IDENTIFIER), "Pass_"+str(_pass))
        model_filename = str(aug_labels_for_this_pass) #+ '_' + '{train_acc:.2f}'

        print("Model directory path: ", model_dirpath)
        print("Model filename: ", model_filename)

        if(not os.path.exists(model_dirpath)):
            os.makedirs(model_dirpath)

        es = EarlyStopping('train_acc', check_finite=True, patience=20, verbose=True, mode="max")
        mc = ModelCheckpoint(monitor = 'train_acc',
                             dirpath = model_dirpath, 
                             filename = model_filename,
                             #auto_insert_metric_name = True,
                             every_n_epochs=1)

        trainer = pl.Trainer(gpus=GPUs, 
                        max_epochs=INNER_EPOCHS,
                        callbacks=[es, mc],
                        log_every_n_steps=22
                        )

        if(AUTO_LR_FIND):
            lr_finder = trainer.tuner.lr_find(model, train_image_loader, update_attr=True)
            new_lr = lr_finder.suggestion()
            print("New suggested learning rate is: ", new_lr)
            model.hparams.learning_rate = new_lr

        # Train on the train set
        trainer.fit(model, train_image_loader)

        # Obtain metrics on the validation set        
        validation_results = trainer.validate(dataloaders=val_image_loader)

        # if(_pass == 1):
        #     run_acc_no_aug = validation_results[0]['val_acc']
        #     run_loss_no_aug = validation_results[0]['val_loss']
        #     run_f1_no_aug = validation_results[0]['f1_score']

    
        run_acc = validation_results[0]['val_acc']
        run_loss = validation_results[0]['val_loss']
        run_f1 = validation_results[0]['f1_score']

        #Add the current augmentation and metric of choice to the results dictionary
        # if(_pass == 1):
        #     _aug_results['aug'] += []
        #     _aug_results['aug_label'] += ['No Aug']
    
        _aug_results['aug'] += [_aug]
        _aug_results['aug_label'] += [_aug_label]

        if(METRIC == 'accuracy'):
            _aug_results['metric'] += [run_acc]
        else:
            _aug_results['metric'] += [run_f1]
        

        _model_paths_dict[_aug_label] = str(os.path.join(model_dirpath, model_filename+'.ckpt'))
        

    #Obtain the augmentation which gave the best metric for this pass

    sorted_dict = {}
    augmentations = _aug_results['aug']
    augmentation_labels = _aug_results['aug_label']
    metric_scores = _aug_results['metric']
    

    #print("acc_scores", acc_scores)
    print("metric used: ", METRIC)
    print("metric scores: ", metric_scores)
    print("augmentation_labels", augmentation_labels)
    print("augmentations", augmentations)

    ######## PERFORMING SORTING OF DICTIONARY ############


    _temp_d = {}
    for i in range(len(metric_scores)):
        _temp_d[metric_scores[i]] = [augmentations[i], augmentation_labels[i]]

    _sorted_metric = sorted(list(_temp_d.keys()), reverse=True)

    _sorted_d = {}
    for i in range(len(_sorted_metric)):
        _sorted_d[_sorted_metric[i]] = _temp_d[_sorted_metric[i]]

    l1 = list(_sorted_d.keys())
    l2 = []
    l3 = []

    _sored_d_values = list(_sorted_d.values())

    for i in range(len(_sorted_d)):
        _ = _sored_d_values[i]
        l2.append(_[0])
        l3.append(_[1])

    sorted_dict['metric'] = l1
    sorted_dict['aug'] = l2
    sorted_dict['aug_label'] = l3

    ###############################################

    _best_augmentation = sorted_dict['aug'][0]
    _best_augmentation_label = sorted_dict['aug_label'][0]
    _best_aug_model_path = _model_paths_dict[_best_augmentation_label]

    print("Best augmentation in this pass: {}".format(_best_augmentation))
    print("Best augmentation model path: {}".format(_best_aug_model_path))
    print("\n")
    print("\n")

    all_selected_augs.append(_best_augmentation)    
    all_selected_augs_labels.append(_best_augmentation_label)
    all_aug_list.remove(_best_augmentation)         
    all_aug_labels_list.remove(_best_augmentation_label)

    #Add these results to validation dictionary to plot intermediate results
    val_results_dict['aug'] += [all_selected_augs[:]]
    val_results_dict['aug_label'] += [all_selected_augs_labels[:]]
    val_results_dict['metric'] += [sorted_dict['metric'][0]]

    # Select the model associated with the best augmentation for testing
    print("################ Initiating testing process #####################")
    test_results = trainer.test(dataloaders=test_image_loader, 
                                ckpt_path=_best_aug_model_path)


    test_acc = test_results[0]['test_acc']
    test_loss = test_results[0]['test_loss']
    test_f1 = test_results[0]['f1_score']

    test_results_dict['aug'] += [all_selected_augs[:]]
    test_results_dict['aug_label'] += [all_selected_augs_labels[:]]

    if(METRIC == 'accuracy'):
        test_results_dict['metric'] += [test_acc]
    else:
        test_results_dict['metric'] += [test_f1]

    #print(" ###### TEST RESULTS FROM THE BEST AUGMENTATION FROM THIS PASS ######")
    #print("Augmentation: ", test_acc)
    #print("Accuracy: ", test_acc)
    #print("Acc: ", test_acc)


######### SORTING test_results_dict ##########

print("############ SORTING ALL THE TEST RESULTS TO OBTAIN FINAL GREEDY SEARCH RESULTS ###########")

sorted_test_results_dict = {}
augmentations = test_results_dict['aug']
augmentations_labels = test_results_dict['aug_label']
metric_scores = test_results_dict['metric']

print("\n")
pprint(test_results_dict)
print("\n")

#l1, l2, l3 = (list(t) for t in zip(*sorted(zip(acc_scores, augmentations, augmentations_labels), reverse=True)))
l1, l2, l3 = (list(t) for t in zip(*sorted(zip(metric_scores, augmentations, augmentations_labels), reverse=True)))


#sorted_test_results_dict['acc'] = l1
sorted_test_results_dict['metric'] = l1
sorted_test_results_dict['aug'] = l2
sorted_test_results_dict['aug_label'] = l3


print("############################ RESULTS FROM VALIDATION STAGE ############################")
pprint(val_results_dict['aug_label'])
pprint(val_results_dict['metric'])
#pprint(val_results_dict['acc'])
#pprint(val_results_dict['f1'])

print("############################ FINAL TEST RESULTS AFTER SORTING ############################")
pprint(sorted_test_results_dict['aug_label'])
pprint(sorted_test_results_dict['metric'])
#pprint(sorted_test_results_dict['acc'])
#pprint(sorted_test_results_dict['f1'])

print("################################### BEST AUGMENTATION AFTER GREEDY SEARCH ##########################")
pprint(sorted_test_results_dict['aug_label'][0])

print("################################### BEST Metric SCORE AFTER GREEDY SEARCH ##########################")
pprint(sorted_test_results_dict['metric'][0])
#pprint(sorted_test_results_dict['acc'][0])
#pprint(sorted_test_results_dict['f1'][0])

######## PLOTTING RESULTS ########

info_dict = {
                 'dataset': DATASET,
                 'encoder': ENCODER,
                 'finetune': DO_FINETUNE,
                 'experiment': EXPERIMENT,
                 'train_mlp': TRAIN_MLP,
                 'use_mean_embeddings': USE_MEAN_EMBEDDINGS,
                 'auto_lr': AUTO_LR_FIND,
                 'metric': METRIC
            }

aug_bit_vector = utils.plot_greedy_augmentations(new_aug_dict, 
                                sorted_test_results_dict,
                                info_dict,
                                save_plot=SAVE_PLOTS)

utils.plot_intermidiate_results(val_results_dict, test_results_dict, 
                                info_dict, save_plot=SAVE_PLOTS)

_similar_augmentations, _similar_aug_labels, _similar_metric = utils.get_best_augmentation_schemes(sorted_test_results_dict)

if(len(_similar_aug_labels) > 1):
    print("Greedy search recommends the following policies ...")
    
    for i in range(len(_similar_aug_labels)):
        _policy = _similar_augmentations[i]
        _policy_labels = _similar_aug_labels[i]
        _metric_val = _similar_metric[i]

        print("Policy: {} | Performance: {}".format(_policy_labels, _metric_val))
        print("\n")

else:
    print("The greedy search recommends only 1 optimal policy.")

end_time = time.time()

print("Augmentation Bit Representation: {}".format(aug_bit_vector))
print("Execution Time (hours): {}".format((end_time - start_time)/3600))
print("Unique Identifier: {}".format(IDENTIFIER))
print("Metric Used: {}".format(METRIC))












        


