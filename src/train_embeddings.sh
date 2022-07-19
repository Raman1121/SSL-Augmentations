#!/bin/bash
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --partition ampere
#SBATCH --gres=gpu:1
#SBATCH --account BMAI-CDT-SL2-GPU
#SBATCH --time=36:00:00

cd /home/co-dutt1/rds/hpc-work/SSL-Augmentations/src
module load cuda/11.2
module load cudnn/8.1_cuda-11.2
source ~/rds/hpc-work/miniconda3/bin/activate mlp

python train_integrated.py --dataset cancer_mnist --model dorsal 

