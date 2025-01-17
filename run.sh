#!/bin/bash

#SBATCH --job-name=docs_ex1
#SBATCH --output=docs_ex1.out
#SBATCH --gpus=2
#SBATCH --partition=ampere
#SBATCH --time=36:00:00         # Hours:Mins:Secs

hostname
nvidia-smi --list-gpus

conda init
conda activate bias
srun python3 1a_bbq_mistral_evaluation.py