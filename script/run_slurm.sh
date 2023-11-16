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
## Uncomment this to run training
python3 ./src/train.py

## Uncomment this to run testing
# python3 ./src/test.py

## Uncomment this to run evaluation
# python3 ./src/evaluate-v2.0.py ./dataset/dev-v1.1.json ./result/prediction/ensemble_all_high_ranking.json

## Uncomment this to run ensemble
# python3 /home/n/njinyuan/CS4248/CS4248_G10/src/ensemble.py

echo "finished training, deactivating env"
conda deactivate
