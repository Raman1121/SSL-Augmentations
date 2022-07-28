# SSL-Augmentations
This repository contains the code to replicate the experiments in XXXX.
<A project to investigate and study the optimal invariances in different Medical Imaging Tasks>

## Introduction
In this work, we present a two simple approaches to find the optimal augmentation policies for different medical image classification tasks. We show that medical imaging tasks require augmentation schemes different than those found on datasets with natural images and further study how our results vary with dataset size.


## Set up the conda environment
1. Use the command `conda env create -f environment.yml` to create an identical conda environment as used in this project. An environment named "mlp" would be created with the required packages.

2. Activate this environment using the command `conda activate mlp`.


## Experiments

Experiments can be run by using the scripts present in the src/ folder. Some required steps before running the experiments -

1. Download the datasets inside the data/ folder.
2. Modify the paths to datasets and associated CSV files in the config files present within the conf/ folder.

### Greedy Search
To search for the augmentation policies following a greedy searching method described in section 3.3 of the report.

```
python greedy_search_single_model.py --dataset <dataset-name> --encoder <encoder-name> --train_mlp <True/False> --do_finetune

```

### Random Search
To search for the augmentation policies following a random searching method described in section 3.3 of the report.

```
python random_search.py --dataset <dataset-name> --encoder <encoder-name> --train_mlp <True/False> --do_finetune

```

Additional parameters: 

1. Provide your learning rate: --learning_rate 0.01
2. Find optimal learning rate automatically: --auto_lr_find True
3. Perform Finetuning: --do_finetune


### Evaluate Dorsal, Ventral, and Default Embeddings

```
python train_integrated.py --dataset <dataset-name> --model <dorsal/ventral/default>
```

### Train using an augmentation scheme generated through Bernoulli sampling

```
python k_bit_embeddings.py --dataset <dataset-name> --num_runs <number of runs>
```