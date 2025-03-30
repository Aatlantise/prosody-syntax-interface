#!/bin/bash

#SBATCH --job-name="duration-syll-syn"
#SBATCH --output="%x.o%j"
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu
#SBATCH --time=12:00:00
#SBATCH --mem=0
#SBATCH --mail-user=jm3743@georgetown.edu
#SBATCH --mail-type=END,FAIL

source env.sh



python src/train.py experiment=syntactic/duration/duration_regression_syll_gpt2 logger=csv >> syll.log
python src/train.py experiment=syntactic/duration/duration_regression_syll_gpt2_np logger=csv >> syll.log
python src/train.py experiment=syntactic/duration/duration_regression_syll_gpt2_npvp logger=csv >> syll.log