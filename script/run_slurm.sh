#!/bin/bash

#SBATCH --time=03:00:00
#SBATCH --job-name=cs4248-g10
#SBATCH --gpus=a100mig:1
#SBATCH --partition=medium
#SBATCH --output=logs/cs4248_%j.slurmlog
#SBATCH --error=logs/cs4248_%j.slurmlog

# Get some output about GPU status before starting the job
nvidia-smi 

## edit the file path
source /home/n/njinyuan/miniconda3/etc/profile.d/conda.sh

## edit the environment
echo "activating environment"
conda activate cs4248
conda info --env

echo "===training==="
python3 /home/n/njinyuan/CS4248/CS4248_G10/src/train.py

echo "finished training, deactivating env"
deactivate
