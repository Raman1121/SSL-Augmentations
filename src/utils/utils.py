from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader
import torchvision.transforms as T

from pl_bolts.models.self_supervised import SimCLR
from pytorch_lightning.core.lightning import LightningModule
from pl_bolts.datasets import DummyDataset
from dataset import retinopathy_dataset

from dataset import retinopathy_dataset, cancer_mnist_dataset, mura_dataset, chexpert_dataset

from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight

import torch
import os
import pandas as pd
import numpy as np
import yaml
import matplotlib.pyplot as plt
import seaborn as sns


def check_data_loader(dl):

    for x, y in dl:
        
        print("\n")
        print("Checking the data loader")
        print("Size of Image Tensor: {}".format(x.size()))
        print("Size of Labels Tensor: {}".format(y.size()))
        print("\n")
        
        break


def CIFAR10_dataset(train_transforms, test_transforms, download_dir='../../data'):

    if(train_transforms == None):
        train_transforms = T.Compose([T.ToTensor()])

    if(test_transforms == None):
        test_transforms = T.Compose([T.ToTensor()])

    train_set = CIFAR10(download_dir, download=True,
                        transform=train_transforms)


    test_set = CIFAR10(download_dir, download=True,
                        transform=test_transforms, train=False)

    return (train_set, test_set)

def get_dummy_dataset(input_shape=(3, 224, 224), label_shape=(1,), num_samples=100):
    dummy_ds = DummyDataset(input_shape, label_shape, num_samples=num_samples)

    return dummy_ds

def get_dataloaders(yaml_data, DATASET, train_transform, basic_transform):

    #Dataset constants
    DATASET_ROOT_PATH = yaml_data['all_datasets'][DATASET]['root_path']
    TRAIN_DF_PATH = yaml_data['all_datasets'][DATASET]['train_df_path']
    VALIDATION_SPLIT = 0.3
    TEST_SPLIT = 0.2
    SEED = 42
    NUM_CLASSES = yaml_data['all_datasets'][DATASET]['num_classes']
    BATCH_SIZE = yaml_data['run']['batch_size']
    SUBSET = yaml_data['run']['subset']

    main_df = pd.read_csv(TRAIN_DF_PATH)

    train_dataset = None
    val_dataset = None
    test_dataset = None

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

        # Creating training and validation splits
        train_df, val_df = train_test_split(main_df, test_size=VALIDATION_SPLIT,
                                    random_state=SEED)
        train_df = train_df.reset_index(drop=True)
        val_df = val_df.reset_index(drop=True)

        #Checking if SUBSET size is greater than the size of the dataset itself.
        TRAIN_SUBSET = len(main_df) if len(main_df) < int(0 if SUBSET == None else SUBSET) else SUBSET
        TEST_SUBSET = len(test_df) if len(test_df) < int(0 if SUBSET == None else SUBSET) else SUBSET
        VAL_SUBSET = len(val_df) if len(val_df) < int(0 if SUBSET == None else SUBSET) else SUBSET

        train_dataset = retinopathy_dataset.RetinopathyDataset(df=main_df, cat_labels_to_include=TRAIN_CAT_LABELS, 
                                                            transforms=train_transform, subset=TRAIN_SUBSET)

        val_dataset = retinopathy_dataset.RetinopathyDataset(df=val_df, cat_labels_to_include=VAL_CAT_LABELS, 
                                                            transforms=train_transform, subset=VAL_SUBSET)

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

        # Creating training, validation, and test splits
        _, val_df = train_test_split(main_df, test_size=VALIDATION_SPLIT,
                                    random_state=SEED)

        _ = _.reset_index(drop=True)
        val_df = val_df.reset_index(drop=True)

        train_df, test_df = train_test_split(_, test_size=TEST_SPLIT,
                                    random_state=SEED)

        train_df = train_df.reset_index(drop=True)
        test_df = test_df.reset_index(drop=True)
        

        #Checking if SUBSET size is greater than the size of the dataset itself.

        TRAIN_SUBSET = len(train_df) if len(train_df) < int(0 if SUBSET == None else SUBSET) else SUBSET
        VAL_SUBSET = len(val_df) if len(val_df) < int(0 if SUBSET == None else SUBSET) else SUBSET
        TEST_SUBSET = len(test_df) if len(test_df) < int(0 if SUBSET == None else SUBSET) else SUBSET

        train_dataset = cancer_mnist_dataset.CancerMNISTDataset(df=train_df, transforms=train_transform, 
                                                                subset=TRAIN_SUBSET)

        val_dataset = cancer_mnist_dataset.CancerMNISTDataset(df=val_df, transforms=train_transform, 
                                                                subset=VAL_SUBSET)

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


    return_dict = {'train_image_loader': train_image_loader,
                    'val_image_loader': val_image_loader,
                    'test_image_loader': test_image_loader,
                    'activation': ACTIVATION,
                    'loss_fn': LOSS_FN,
                    'multilable': MULTILABLE,
                    'class_weights': CLASS_WEIGHTS
                    }
    
    return return_dict

def plot_run_stats(all_acc, all_loss, info_dict, save_dir='saved_plots/', save_plot=True):
    if(save_plot):

        num_runs = info_dict['num_runs']
        dataset = info_dict['dataset']
        encoder = info_dict['encoder']
        finetune = info_dict['finetune']
        experiment = info_dict['experiment']

        runs = list(range(num_runs))

        plt.plot(runs, all_acc, label='Test Accuracy', linewidth=4)
        plt.plot(runs, all_loss, label='Test Loss', linewidth=4)
        plt.plot(runs, [np.mean(all_acc)]*len(all_acc), label='Mean Acc', linewidth=2, linestyle='dashed')

        plt.xticks(np.arange(num_runs), runs)
        plt.xlabel('Runs')
        plt.ylabel('Value')

        plt.suptitle("Test Accuracy and Loss across different runs for {} dataset".format(dataset))
        plt.title("Encoder: {} | Finetune: {} | Experiment: {}".format(encoder, str(finetune), experiment), fontsize=10)
        #title = ' '.join(str(i) for i in aug_bit)
        #title = '['+title+']'
        #prefix = 'Test Acc and Loss'
        #plt.title(prefix)

        plt.legend()
        plt.show()

        #save_path = os.path.join(save_dir, dataset+'.png')

        plt.savefig(save_dir + experiment + '_' + dataset +'_' + str(num_runs) + '_' + encoder + '.png')
            
        
    else:
        print("Saving plot skipped.")


def plot_greedy_augmentations(aug_dict, aug_dict_labels, sorted_test_results_dict, 
                              info_dict, save_dir='saved_plots/', save_plot=True):

    dataset = info_dict['dataset']
    encoder = info_dict['encoder']
    finetune = info_dict['finetune']
    experiment = info_dict['experiment']
    
    best_aug_list = sorted_test_results_dict['aug'][0]
    best_aug_labels = sorted_test_results_dict['aug_label'][0]

    include_vector = [0]*len(aug_dict)

    for aug in list(aug_dict_labels.keys()):
        if(aug in best_aug_labels):
            index = aug_dict_labels[aug] - 1
            include_vector[index] = 1

    d = {'Augmentations': list(aug_dict_labels.keys()),
         'Present': include_vector}
    df = pd.DataFrame(d)

    #plot = sns.heatmap(df, vmin=0, vmax=1, cbar=False, cmap="winter")
    #fig = plot.get_figure()

    # if(save_plot):
    #     fig.savefig(save_dir + experiment + '_' + dataset + '_' + encoder + '.png')

    return include_vector


def gen_binomial_dict(aug_dict):
    '''
    A function to obtain augmentations through binomial sampling
    '''

    temp_dict = {}
    for k, v in aug_dict.items():
        temp_dict[v] = k

    aug_bit = []
    for i in range(len(aug_dict)):
        s = np.random.binomial(1, 0.5)          #Each augmentation has a probability of 0.5
        aug_bit.append(s)

    #Finding indexes where aug_bit is 1
    result = np.where(np.array(aug_bit) == 1)
    indexes = result[0] + 1

    aug_list = []
    for index in indexes:
        aug_list.append(temp_dict[index])

    return aug_list, aug_bit


def sort_dictionary(results_dict):
    
    all_acc = results_dict['acc']
    all_loss = results_dict['loss']
    all_f1 = results_dict['f1']
    all_repr = results_dict['k_bit_representation']
    all_runs = results_dict['run']

    l1, l2, l3, l4, l5 = (list(t) for t in zip(*sorted(zip(all_acc, all_loss, all_f1, all_repr, all_runs), reverse=True)))

    sorted_dict = {}
    sorted_dict['acc'] = l1
    sorted_dict['loss'] = l2
    sorted_dict['f1'] = l3
    sorted_dict['k_bit_representation'] = l4
    sorted_dict['run'] = l5

    return sorted_dict



def run_one_aug(dl, encoder, aug_dict, num_samples, num_aug_samples=10):

    '''
        * Feed a dataset of images and obtain embeddings for all images. 
        * Final output for a dataset should have size [num_samples, embedding_size]
        * 'embedding_size' depends on the encoder used. [ResNet, SimCLR = 2048, ViTs = 768]
    '''

    if(dl == None):
        raise AssertionError("Input for Dataloader (dl) is required")
    if(encoder == None):
        raise AssertionError("Input for encoder is required")
    if(aug_dict == None):
        raise AssertionError("Input for augmentation dictionary (aug_dict) is required")
    if(num_samples == None):
        raise AssertionError("Input for number of samples in the dataset (num_samples) is required")

    all_embeddings = torch.tensor([])   #A tensor to hold mean embeddings of all images in a dataset
    final_dataset_embeddings = torch.tensor([]) #A tensor holding mean embeddings for all images in a dataset but reshaped to correct dimensions

    for x, y in dl:
    
        '''
        LOOP NUMBER 1
        For each image:
        [Selecting an image from the dataset]

        '''
        for i in range(x.size()[0]):
            all_embed_img = torch.tensor([])    #A tensor to hold embeddings of all augmentations of an image

            '''
            LOOP NUMBER 2
            For each augmentation:
            [Selecting an augmentation from the set of all augmentations]

            '''
            for j in range(len(aug_dict)):
                aug = list(aug_dict.values())[j]
                all_embed_aug = torch.tensor([])    #A tensor to hold all embeddings for different samples of a particular augmentation

                #print("Augmentation: ", aug)

                '''
                LOOP NUMBER 3
                FOR EACH SAMPLE OF AN AUGMENTATION
                [Sampling an augmentation]
                '''
                for k in range(num_aug_samples):
                    
                    #print("Sample number for this augmentation: ", k)
                    preprocess = T.Compose([T.ToPILImage(), aug, T.ToTensor()])
                    aug_img = preprocess(x[i])
                    embedding = encoder(aug_img.unsqueeze(0))

                    all_embed_aug = torch.cat((all_embed_aug, embedding), 0)

                    #print("Size of the tensor holding all samples for this augmentation: ", all_embed_aug.size())
                
                all_embed_aug = torch.mean(all_embed_aug, 0)    #Mean of different samples of the same augmentation [1, embedding_size]

                all_embed_img = torch.cat((all_embed_img,       #Adding the mean embedding obtained via different samples of an augmentation.
                                            all_embed_aug), 0)

            all_embed_img = torch.reshape(all_embed_img, (len(aug_dict), embedding.size()[1]))   #[number of augmentations, embedding_size]
            all_embed_img = torch.mean(all_embed_img, 0) #Final embeddings generated after multiple samples of all augmentations for an image [1, embedding_size]
            
            all_embeddings = torch.cat((all_embeddings,         #Adding mean embedding for an image to the list of all embeddings for a dataset.
                                        all_embed_img), 0)

    print("SHAPE OF ALL EMBEDDINGS FOR ALL IMAGES", all_embeddings.size()) #[num_samples, 768]
    print('\n')

    final_dataset_embeddings = torch.reshape(all_embeddings, (num_samples, embedding.size()[1])) #Final reshaping for all images in a dataset.

    return final_dataset_embeddings
    


    