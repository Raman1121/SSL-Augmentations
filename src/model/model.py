import torch
from torch import nn

import timm
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform


import torchvision.models as models
from pl_bolts.models.self_supervised import SimCLR
from pytorch_lightning.core.lightning import LightningModule

class Encoder(LightningModule):
    def __init__(self, encoder='resnet50_supervised'):
        super().__init__()
        
        self.encoder = encoder

        #TODO: List of encoders in the configuration file
        if encoder not in ['resnet50_supervised', 'simclr_r50', 'vit_base_patch16_224_in21k', 'vit_base_patch32_224_in21k']:
            raise AssertionError("Encoder not in the list of supported encoders.")
        
        
        if(self.encoder == 'resnet50_supervised'):
            backbone = models.resnet50(pretrained=True)
            num_filters = backbone.fc.in_features
            layers = list(backbone.children())[:-1]
            self.feature_extractor = nn.Sequential(*layers)


        elif(self.encoder == 'simclr_r50'):
            weight_path = 'https://pl-bolts-weights.s3.us-east-2.amazonaws.com/simclr/bolts_simclr_imagenet/simclr_imagenet.ckpt'
            simclr = SimCLR.load_from_checkpoint(weight_path, strict=False)
            self.feature_extractor = simclr.encoder

        elif(self.encoder == 'vit_base_patch16_224_in21k'):
            self.feature_extractor = timm.create_model('vit_base_patch16_224_in21k', pretrained=True, num_classes=0)
            config = resolve_data_config({}, model=self.feature_extractor)
            transform = create_transform(**config)

        elif(self.encoder == 'vit_base_patch32_224_in21k'):
            self.feature_extractor = timm.create_model('vit_base_patch32_224_in21k', pretrained=True, num_classes=0)
            config = resolve_data_config({}, model=self.feature_extractor)
            transform = create_transform(**config)
            

    def forward(self, x):
        self.feature_extractor.eval()
        #batch_size, channels, height, width = x.size()

        with torch.no_grad():
            if(self.encoder == 'resnet50_supervised'):
                representations = self.feature_extractor(x).flatten(1)
            elif(self.encoder == 'simclr_r50'):
                representations = self.feature_extractor(x)[0]
            elif(self.encoder == 'vit_base_patch16_224_in21k' or self.encoder == 'vit_base_patch32_224_in21k'):
                representations = self.feature_extractor(x)
            

        return representations