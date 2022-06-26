import random

import albumentations as A
from albumentations.augmentations.transforms import *
from albumentations.augmentations.crops.transforms import *
from albumentations.augmentations.geometric.rotate import *
from albumentations.augmentations.geometric.transforms import *
from albumentations.augmentations.geometric.resize import Resize

class StandardAugmentations:

    def __init__(self, shuffle = False):

        self.aug_dict_labels = {
                                #'CLAHE': CLAHE(),
                                'CJ': ColorJitter(),
                                'DS': Downscale(),
                                'EB': Emboss(),
                                'SSR': ShiftScaleRotate(),
                                'HF': HorizontalFlip(),
                                'VF': VerticalFlip(),
                                'IC': ImageCompression(),
                                'Rotate': Rotate(),
                                'INet_Norm':Normalize(),
                                'Perspective':Perspective()
                                } 

        self.aug_dict = {
                        #CLAHE(): 1,
                        ColorJitter(): 2,
                        Downscale(): 3,
                        Emboss(): 4,
                        ShiftScaleRotate(): 5,
                        HorizontalFlip(): 6,
                        VerticalFlip(): 7,
                        ImageCompression(): 8,
                        Rotate(): 9,
                        Normalize(): 10,
                        Perspective(): 11
                        }

        self.new_aug_dict = {   
                                'Equalize': Equalize(),
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
    
    