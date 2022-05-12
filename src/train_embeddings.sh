#!/bin/bash
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --partition ampere
#SBATCH --gres=gpu:1
#SBATCH --account BMAI-CDT-SL2-GPU
#SBATCH --time=7:00:00

python train_integrated.py --dataset mura --model default 

