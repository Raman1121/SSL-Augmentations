from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as T
from torchvision.io import read_image
import pytorch_lightning as pl

import yaml

class RetinopathyDataset(Dataset):
    def __init__(self, df, transforms=None):
        """
            Parameters
            ----------
            df : pandas.core.frame.DataFrame
                Dataframe containing image paths ['image'], retinopathy level ['level'], and image quality scores ['score']
            transforms : torchvision.transforms.transforms.Compose, default: None
                A list of torchvision transformers to be applied to the training images.
        """

        self.df = df
        self.transforms = transforms

    def __len__(self,):
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
            image = self.transforms(image)

        return image, label


class CheXpertDataset(Dataset):
    def __init__(self, df, transforms):
        self.df = df
        self.transforms = transforms
    
    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        image = read_image(self.df['Path'][idx])
        #label = ....

        if(self.transforms):
            image = self.transforms(image)

        return image, label