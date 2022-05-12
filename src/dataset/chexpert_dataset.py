from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as T
from torchvision.io import read_image, ImageReadMode
import pytorch_lightning as pl

import random

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

        image = read_image(self.df['Path'][idx], mode=ImageReadMode.RGB)
        label = self.df['Pathology'][idx]

        if(self.transforms):
            image = self.transforms(image)

        return image, label
