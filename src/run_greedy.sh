#!/bin/bash
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --partition ampere
#SBATCH --gres=gpu:1
#SBATCH --account BMAI-CDT-SL2-GPU
#SBATCH --time=30:00:00

python greedy_search_augmentations.py --dataset cancer_mnist --encoder 'resnet50' --lr 0.005 --pretrained True --train_mlp True --do_finetune

