from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader
import torchvision.transforms as T

from pl_bolts.models.self_supervised import SimCLR
from pytorch_lightning.core.lightning import LightningModule
from pl_bolts.datasets import DummyDataset

