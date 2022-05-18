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

class ChexpertDataset(Dataset):
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
        self.image_paths = []
        self.image_labels = []

        if(self.subset != None):
            print("Creating a subset of {} samples".format(self.subset))
            
            subset_list = random.sample(range(len(self.df)), self.subset)
            self.df = self.df.iloc[subset_list].reset_index(drop=True)

        else:
            print("Creating the entire dataset.")


        #Preparing labels list
        for index, row in self.df.iterrows():
            img_path = row.Path
            self.image_paths.append(img_path)

            if(len(row) < 14):
                labels = [0]*14
            else:
                labels = []
                for col in row[5:]:
                    if(col == 1):
                        labels.append(1)
                    else:
                        labels.append(0)

            self.image_labels.append(labels)

    def __len__(self):

        """
            Returns
            -------

            Number of samples in our dataset.
        """

        return len(self.image_paths)

    def __getitem__(self, idx):

        """
            Parameters
            ----------
            idx: index to identify a sample in the dataset

            Returns
            -------
            An image and a label from the dataset based on the given index idx.
        """

        image_path = self.image_paths[idx]

        if(self.transforms):

            #Check if torchvision transforms are provided
            if(type(self.transforms) == torchvision.transforms.transforms.Compose):
                image = read_image(image_path, mode=ImageReadMode.RGB)
                image = self.transforms(image)

            #Check if albumentation transforms are provided
            elif(type(self.transforms) == A.core.composition.Compose):
                pillow_image = Image.open(image_path)
                pillow_image = pillow_image.convert('RGB')
                image = np.array(pillow_image)

                #image = image.cpu().detach().numpy()            #Albumentation takes image as a numpy array.
                image = self.transforms(image=image)['image']

                image = torch.from_numpy(image)
                image = image.permute(2, 0, 1)

                #image = image.type(torch.FloatTensor)
                image = image.float()
                #print(image.dtype)

        return image, torch.FloatTensor(self.image_labels[idx])