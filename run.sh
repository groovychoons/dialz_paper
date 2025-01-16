#!/bin/bash

#SBATCH --job-name=docs_ex1
#SBATCH --output=docs_ex1.out
#SBATCH --gpus=1
#SBATCH --partition=hopper
#SBATCH --time=27:00:00         # Hours:Mins:Secs

hostname
nvidia-smi --list-gpus

conda init
conda activate bias
srun python3 1a_bbq_mistral_evaluation.py