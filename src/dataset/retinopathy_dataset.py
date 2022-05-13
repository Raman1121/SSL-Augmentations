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


class RetinopathyDataset(Dataset):
    def __init__(self, df, transforms=None, cat_labels_to_include=['Good', 'Usable', 'Reject'], subset=None):

        """
            Parameters
            ----------
            df : pandas.core.frame.DataFrame
                Dataframe containing image paths ['image'], retinopathy level ['level'], and image quality scores ['continuous_score']
            transforms : torchvision.transforms.transforms.Compose, default: None
                A list of torchvision transformers to be applied to the training images.
            cat_labels_to_include : list, default: ['Good', 'Usable', 'Reject']
                A list of categorical labels to be included in our dataset
        """

        self.df = df
        self.transforms = transforms
        self.cat_labels = cat_labels_to_include
        self.subset = subset

        if(self.subset != None):
            print("Creating a subset of {} samples".format(self.subset))
            
            subset_list = random.sample(range(len(self.df)), self.subset)
            self.df = self.df.iloc[subset_list].reset_index(drop=True)

        else:
            print("Creating the entire dataset.")

        self.df = self.df.loc[self.df['quality'].isin(self.cat_labels)].reset_index(drop=True)

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
        image = read_image(self.df['image'][idx])
        label = self.df['level'][idx]

        if(self.transforms):

            #Check if torchvision transforms are provided
            if(type(self.transforms) == torchvision.transforms.transforms.Compose):
                image = self.transforms(image)

            #Check if albumentation transforms are provided
            elif(type(self.transforms) == A.core.composition.Compose):
                pillow_image = Image.open(self.df['image'][idx])
                image = np.array(pillow_image)

                #image = image.cpu().detach().numpy()            #Albumentation takes image as a numpy array.
                image = self.transforms(image=image)['image']

                image = torch.from_numpy(image)
                image = image.permute(2, 0, 1)

                #image = image.type(torch.FloatTensor)
                image = image.float()
                #print(image.dtype)

        return image, label


class LightningRetinopathyDataset(pl.LightningDataModule):
    def __init__(self, train_dataset, val_dataset, test_dataset, batch_size=32):
        super().__init__()

        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.test_dataset = test_dataset
        self.batch_size = batch_size

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, num_workers=12, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=12, shuffle=False)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, num_workers=12, shuffle=False)
    