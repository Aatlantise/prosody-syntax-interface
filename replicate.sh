#!/bin/bash

#SBATCH --job-name="prosody-redundance-replication"
#SBATCH --output="%x.o%j"
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu
#SBATCH --time=12:00:00
#SBATCH --mem=0
#SBATCH --mail-user=jm3743@georgetown.edu
#SBATCH --mail-type=END,FAIL

source env.sh

python src/train.py experiment=emnlp/camera/prominence_regression_absolute_gpt2_mle logger=csv
python src/train.py experiment=emnlp/finetuning/energy_regression_gpt2 logger=csv
python src/train.py experiment=emnlp/finetuning/pause_regression_after_gpt2 logger=csv