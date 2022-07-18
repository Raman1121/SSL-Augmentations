import random

import albumentations as A
from albumentations.augmentations.transforms import *
from albumentations.augmentations.crops.transforms import *
from albumentations.augmentations.geometric.rotate import *
from albumentations.augmentations.geometric.transforms import *
from albumentations.augmentations.geometric.resize import Resize

class StandardAugmentations:

    def __init__(self, shuffle = False):

        self.new_aug_dict = {   
                                #'Equalize': Equalize(mode='pil'),
                                'CJ': ColorJitter(),
                                'CF': ChannelShuffle(),
                                'GB': GaussianBlur(),
                                'GN': GaussNoise(),
                                'RB': RandomBrightness(),
                                'RC': RandomContrast(),
                                'RGBS': RGBShift(),
                                'Sharpen': Sharpen(),
                                'Affine': Affine(),
                                'HF': HorizontalFlip(),
                                'VF': VerticalFlip(),
                                'Rotate': Rotate(),
                                'INet_Norm': Normalize()
                            }


        if(shuffle):
            self.new_aug_dict = self.shuffle_dictionary(self.new_aug_dict)

    def shuffle_dictionary(self, dictionary):
        _keys = list(dictionary.keys())
        _values = list(dictionary.values())

        random.shuffle(_keys)

        shuffled_dict = {}

        for i in range(len(_keys)):
            shuffled_dict[_keys[i]] = dictionary[_keys[i]]

        return shuffled_dict
    
    