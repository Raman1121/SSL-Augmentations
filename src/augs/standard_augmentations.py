import random

import albumentations as A
from albumentations.augmentations.transforms import *
from albumentations.augmentations.crops.transforms import *
from albumentations.augmentations.geometric.rotate import *
from albumentations.augmentations.geometric.transforms import *
from albumentations.augmentations.geometric.resize import Resize

class StandardAugmentations:

    def __init__(self, shuffle = False, experimental_run = False):

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

        if(experimental_run):
            self.new_aug_dict = self.create_experimental_dict(self.new_aug_dict)

    def shuffle_dictionary(self, dictionary):
        _keys = list(dictionary.keys())
        _values = list(dictionary.values())

        random.shuffle(_keys)

        shuffled_dict = {}

        for i in range(len(_keys)):
            shuffled_dict[_keys[i]] = dictionary[_keys[i]]

        return shuffled_dict

    def create_experimental_dict(self, dictionary):
        _keys = list(dictionary.keys())
        _values = list(dictionary.values())

        experimental_dict = {}

        res = random.sample(list(self.new_aug_dict.items()), k=3)

        for i in res:
            i = list(i)
            _key = i[0]
            _val = i[1]

            experimental_dict[_key] = _val

        return experimental_dict





    