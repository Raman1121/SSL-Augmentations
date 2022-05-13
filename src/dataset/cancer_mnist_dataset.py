import torch
from torch.utils.data import Dataset, DataLoader
import torchvision
from torchvision import transforms as T
from torchvision.io import read_image
import pytorch_lightning as pl
import albumentations as A

import numpy as np
import random
import yaml
from PIL import Image

class CancerMNISTDataset(Dataset):
    def __init__(self, df, transforms=None, subset=None):

        """
            Parameters
            ----------
            df : pandas.core.frame.DataFrame
                Dataframe containing image paths ['image'], retinopathy level ['level'], and image quality scores ['continuous_score']
            transforms : torchvision.transforms.transforms.Compose, default: None
                A list of torchvision transformers to be applied to the training images.
        """

        self.df = df
        self.transforms = transforms
        self.subset = subset

        if(self.subset != None):
            print("Creating a subset of {} samples".format(self.subset))
            
            subset_list = random.sample(range(len(self.df)), self.subset)
            self.df = self.df.iloc[subset_list].reset_index(drop=True)

        else:
            print("Creating the entire dataset.")


    def __len__(self):

        """
            Returns
            -------

            Number of samples in our dataset.
        """

        return len(self.df)

    def __getitem__(self, idx):

        """
            Parameters
            ----------
            idx: index to identify a sample in the dataset

            Returns
            -------
            An image and a label from the dataset based on the given index idx.
        """

        label = self.df['cell_type_idx'][idx]

        if(self.transforms):
            #Check if torchvision transforms are provided
            if(type(self.transforms) == torchvision.transforms.transforms.Compose):
                image = read_image(self.df['path'][idx])
                image = self.transforms(image)

            #Check if albumentation transforms are provided
            elif(type(self.transforms) == A.core.composition.Compose):
                pillow_image = Image.open(self.df['path'][idx])
                image = np.array(pillow_image)

                #image = image.cpu().detach().numpy()            #Albumentation takes image as a numpy array.
                image = self.transforms(image=image)['image']

                image = torch.from_numpy(image)
                image = image.permute(2, 0, 1)
                
                image = image.float()

        return image, label