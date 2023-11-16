#!/bin/bash

#SBATCH --time=24:00:00
#SBATCH --job-name=cs4248-g10
#SBATCH --gpus=titanv:1
#SBATCH --partition=long
#SBATCH --output=logs/cs4248_%j.slurmlog
#SBATCH --error=logs/cs4248_%j.slurmlog

# Get some output about GPU status before starting the job
nvidia-smi 

## edit the file path to your conda file env filepath
source /home/n/njinyuan/miniconda3/etc/profile.d/conda.sh

## edit the environment to your conda environment
echo "activating environment"
conda activate cs4248
conda info --env

echo "===running ensemble==="
python3 ./src/ensemble.py

echo "finished training, deactivating env"
conda deactivate
