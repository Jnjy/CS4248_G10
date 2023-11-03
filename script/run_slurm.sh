#!/bin/bash

#SBATCH --time=24:00:00
#SBATCH --job-name=cs4248-g10
#SBATCH --gpus=titanv:1
#SBATCH --partition=long
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
python3 /home/n/njinyuan/CS4248/CS4248_G10/src/test.py

python3 /home/n/njinyuan/CS4248/CS4248_G10/src/evaluate-g10.py
# python3 /home/n/njinyuan/CS4248/CS4248_G10/src/evaluate-v2.0.py
# /home/n/njinyuan/CS4248/CS4248_G10/result/predictions.json 

echo "finished training, deactivating env"
conda deactivate
