from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader
import torchvision.transforms as T
import random
from PIL import ImageFilter


from pl_bolts.models.self_supervised import SimCLR
from pytorch_lightning.core.lightning import LightningModule
from pl_bolts.datasets import DummyDataset

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

'''
Creating custom Gaussian Blur and Normalize augmentations for Dorsal-Ventral model
'''

class GaussianBlur(object):
    """Gaussian blur augmentation in SimCLR https://arxiv.org/abs/2002.05709"""

    def __init__(self, sigma=[.1, 2.]):
        self.sigma = sigma

    def __call__(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x

class UnNormalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
        Returns:
            Tensor: Normalized image.
        """
        for t, m, s in zip(tensor, self.mean, self.std):
            t.mul_(s).add_(m)
            # The normalize code -> t.sub_(m).div_(s)
        return tensor