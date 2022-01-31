from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader
import torchvision.transforms as T

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

def get_dummy_dataset(input_shape=(224, 224, 3), label_shape=(1,), num_samples=100):
    dummy_ds = DummyDataset(input_shape, label_shape, num_samples=num_samples)

    return dummy_ds