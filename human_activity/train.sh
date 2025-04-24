#!/bin/bash -l

# Slurm parameters
#SBATCH --job-name=JobNameTest
#SBATCH --output=JobNameTest-%j.%N.out
#SBATCH --time=1-00:00:00
#SBATCH --gpus=1

# Activate everything you need
module load cuda/11.8
# Run your python code
python main.py --train LSTM_model
