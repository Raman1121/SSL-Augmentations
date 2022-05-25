from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader
import torchvision.transforms as T

from pl_bolts.models.self_supervised import SimCLR
from pytorch_lightning.core.lightning import LightningModule
from pl_bolts.datasets import DummyDataset
from dataset import retinopathy_dataset

import torch
import os
import pandas as pd
import numpy as np
import yaml
import matplotlib.pyplot as plt

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

def load_DR_dataset(yaml_data, train_transforms, test_transforms):

    ROOT_PATH = yaml_data['DR_DATASET']['root_path']
    DR_TRAIN_DF_PATH = yaml_data['DR_DATASET']['train_df_path']
    DR_TEST_DF_PATH = yaml_data['DR_DATASET']['test_df_path']

    train_df = pd.read_csv(DR_TRAIN_DF_PATH)
    test_df = pd.read_csv(DR_TEST_DF_PATH)

    train_df['image'] = train_df['image'].apply(lambda x: str(ROOT_PATH+'final_train/train/'+x))
    test_df['image'] = test_df['image'].apply(lambda x: str(ROOT_PATH+'final_test/test/'+x))

    train_dataset = retinopathy_dataset.RetinopathyDataset(df=train_df, transforms=train_transforms)
    test_dataset = retinopathy_dataset.RetinopathyDataset(df=test_df, transforms=test_transforms)
    
    return train_dataset, test_dataset

def load_chexpert_dataset(yaml_data, train_transforms, test_transforms):
    ROOT_PATH = yaml_data['CHEX_DATASET']['root_path']
    CHEX_TRAIN_DF_PATH = yaml_data['CHEX_DATASET']['train_df_path']
    CHEX_VALID_DF_PATH = yaml_data['CHEX_DATASET']['valid_df_path']

    train_df = pd.read_csv(CHEX_TRAIN_DF_PATH)
    valid_df = pd.read_csv(CHEX_VALID_DF_PATH)

    train_df['Path'] = train_df['Path'].apply(lambda x: str(ROOT_PATH+x))
    valid_df['Path'] = valid_df['Path'].apply(lambda x: str(ROOT_PATH+x))

    pass

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
    


    