import torch
from torch.utils.data import DataLoader

import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import RichProgressBar, ModelCheckpoint
from pl_bolts.callbacks import PrintTableMetricsCallback

from model import logReg_PL
from dataset import embeddings_dataset

import os
import pandas as pd
import numpy as np


root_path = '/home/co-dutt1/rds/hpc-work/SSL-Augmentations/Embeddings'
dataset = 'retinopathy'
model = 'dorsal'
train_df = pd.read_csv('/home/co-dutt1/rds/hpc-work/MLP-RIQA/Train_set_RIQA_DR_Labels.csv')
test_df = pd.read_csv('/home/co-dutt1/rds/hpc-work/MLP-RIQA/Test_set_RIQA_DR_Labels.csv')

train_folder_path = os.path.join(root_path, dataset, 'train', model)
test_folder_path = os.path.join(root_path, dataset, 'test', model)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

BATCH_SIZE = 256
lr_rate = 0.1
EPOCHS = 100
NUM_CLASSES = 5
EMBEDDINGS_DIM = 2048

train_dataset = embeddings_dataset.EmbeddingsDataset(train_folder_path, train_df, subset=None)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=12)

test_dataset = embeddings_dataset.EmbeddingsDataset(test_folder_path, test_df, subset=None)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=12)

print(len(train_dataset))

logreg_model_pl = logReg_PL.LogisticRegression(input_dim=EMBEDDINGS_DIM, output_dim=NUM_CLASSES, 
                                                learning_rate = lr_rate, batch_size=BATCH_SIZE, 
                                                lr_scheduler='reduce_plateau')

trainer = pl.Trainer(gpus=1, 
                    max_epochs=EPOCHS)

#lr_finder = trainer.tuner.lr_find(logreg_model_pl, train_loader)
#new_lr = lr_finder.suggestion()
#print("New suggested learning rate is: ", new_lr)
#logreg_model_pl.hparams.learning_rate = new_lr

trainer.fit(logreg_model_pl, train_loader)
trainer.test(dataloaders=test_loader)


