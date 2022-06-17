import sys
sys.path.append('/home/co-dutt1/rds/hpc-work/SSL-Augmentations/src')

from utils import utils
import pytest
import os
from pprint import pprint
import yaml

from tests.conftest import *

import albumentations as A
from albumentations.augmentations.transforms import *
from albumentations.augmentations.crops.transforms import *
from albumentations.augmentations.geometric.rotate import *
from albumentations.augmentations.geometric.transforms import *
from albumentations.augmentations.geometric.resize import Resize

@pytest.mark.parametrize(
    "aug_bit_vector",
    [
        [1,1,1,1,1,1,1,1,1,1,1],
        [0,0,0,0,0,0,0,0,0,0,0],

    ]
)
def test_get_aug_from_vector(aug_bit_vector):

    aug_dict_labels = {
                   'CLAHE': 1,
                   'CJ': 2,
                   'DS': 3,
                   'EB': 4,
                   'SSR': 5,
                   'HF': 6,
                   'VF': 7,
                   'IC': 8,
                   'Rotate': 9,
                   'INet_Norm':10,
                   'Perspective':11
                   }

    _selected_augs = utils.get_aug_from_vector(aug_dict_labels, aug_bit_vector)

    if(aug_bit_vector == [1,1,1,1,1,1,1,1,1,1,1]):
        assert _selected_augs == list(aug_dict_labels.keys())

    if(aug_bit_vector == [0,0,0,0,0,0,0,0,0,0,0]):
        assert _selected_augs == []

