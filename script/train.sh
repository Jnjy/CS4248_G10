#!/bin/bash

## edit the file path
source /home/n/njinyuan/miniconda3/etc/profile.d/conda.sh

## edit the environment
echo "activating environment"
conda activate cs4248

echo "===training==="
python3 /home/n/njinyuan/CS4248/CS4248_G10/src/train.py

echo "finished training, deactivating env"
deactivate