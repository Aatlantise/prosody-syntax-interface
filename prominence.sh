#!/bin/bash

#SBATCH --job-name="syntax-prosody-interface"
#SBATCH --output="%x.o%j"
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu
#SBATCH --time=12:00:00
#SBATCH --mem=0
#SBATCH --mail-user=jm3743@georgetown.edu
#SBATCH --mail-type=END,FAIL

source env.sh

python src/train.py experiment=syntactic/finetuning/prominence_regression_relative_gpt2 logger=csv seed=1
python src/train.py experiment=syntactic/finetuning/prominence_regression_relative_gpt2 logger=csv seed=2
python src/train.py experiment=syntactic/finetuning/prominence_regression_relative_gpt2 logger=csv seed=3