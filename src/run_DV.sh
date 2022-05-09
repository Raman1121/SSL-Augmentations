#!/bin/bash
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --partition ampere
#SBATCH --gres=gpu:1
#SBATCH --account BMAI-CDT-SL2-GPU
#SBATCH --time=5:00:00

python create_DV_embeddings.py --create_embd_for train --model dorsal --remove_existing_dataset True 


