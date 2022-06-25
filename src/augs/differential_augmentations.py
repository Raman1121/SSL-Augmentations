import kornia as K

import torchvision.transforms as transforms
import torch
import torch.nn as nn

import os
from PIL import Image
import numpy as np
import pandas as pd


'''
This file implements the standard augmentations but with learnable parameters
'''

class DiffAugBaseClass(nn.Module):
    def __init__(self):
    super(DiffAugBaseClass, self).__init__()

    self.tensor_fn = lambda x: torch.tensor(x, dtype=torch.float64)
    self.param_fn = lambda x: nn.Parameter(self.tensor_fn(x))

class DiffColorJtter(DiffAugBaseClass):

    '''

    Differential version of the ColorJitter augmentation (https://pytorch.org/vision/main/generated/torchvision.transforms.ColorJitter.html) 
    with learnable parameters for brightness, contrast, hue, and saturation.

    '''
    def __init__(self, b, c, h, s):
        super(DiffColorJtter, self).__init__()

        self.brightness = self.param_fn([b, b])
        self.contrast = self.param_fn([c, c])
        self.hue = self.param_fn([h, h])
        self.saturation = self.param_fn([s, s])

        self.brightness_limit = [0, 1]
        self.contrast_limit = [0, 1]
        self.hue_limit = [-0.5, 0.5]
        self.saturation_limit = [0, 1]

        self.jitter = K.augmentation.ColorJitter(self.brightness, self.contrast, 
                                    self.saturation, self.hue, p=0.5)   
    
    def forward(self, _input):

        _input= self.jitter(_input)
        return _input



class DiffRotation(DiffAugBaseClass):
    '''

    Differential version of the Rotation augmentation (https://pytorch.org/vision/stable/generated/torchvision.transforms.RandomRotation.html#torchvision.transforms.RandomRotation) 
    with learnable parameters for the rotation angle.

    '''

    def __init__(self, angle):
        super(DiffRotation, self).__init__()

        self.rotation_angle = self.param_fn([angle, angle])
        self.random_rotation = K.augmentation.RandomRotation(degrees = self.rotation_angle)

        self.rotation_angle_limit = 

    def forward(self, _input):

        _input = self.random_rotation(_input)
        return _input



class DiffCLAHE(DiffAugBaseClass):

    def __init__(self, clip_value):
        super(DiffCLAHE, self).__init__()

        self.clip_value = clip_value
        self.clahe = K.enhance.equalize_clahe(clip_value=self.clip_value)

    def forward(self, _input):

        _input = self.clahe(_input)
        return _input

class DiffNormalize(DiffAugBaseClass):

    '''

    Differential version of the Normalize augmentation (https://pytorch.org/vision/main/generated/torchvision.transforms.Normalize.html) 
    with learnable parameters for the channel means and standard deviations.

    '''

    def __init__(self, channel_means, channel_stds):
        super(DiffNormalize, self).__init__()

        assert isinstance(channel_means, tuple) == True
        assert isinstance(channel_stds, tuple) == True

        self.channel_means = channel_means
        self.channel_stds = channel_stds

        self.normalize_aug = K.augmentation.Normalize(mean = self.channel_means, 
                                                      std = self.channel_stds)

    def forward(self, _input):

        _input = self.normalize_aug(_input)
        return _input