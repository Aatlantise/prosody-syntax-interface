#!/bin/bash

#SBATCH --job-name="prominence-rel"
#SBATCH --output="%x.o%j"
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu
#SBATCH --time=12:00:00
#SBATCH --mem=0
#SBATCH --mail-user=jm3743@georgetown.edu
#SBATCH --mail-type=END,FAIL

source env.sh

python src/train.py experiment=syntactic/prominence/prominence_regression_relative_gpt2 logger=csv seed=1 >> prom-rel.log
python src/train.py experiment=syntactic/prominence/prominence_regression_relative_gpt2_np logger=csv seed=1 >> prom-rel.log
python src/train.py experiment=syntactic/prominence/prominence_regression_relative_gpt2_npvp logger=csv seed=1 >> prom-rel.log


python src/train.py experiment=syntactic/prominence/prominence_regression_relative_gpt2 logger=csv seed=2 >> prom-rel.log
python src/train.py experiment=syntactic/prominence/prominence_regression_relative_gpt2_np logger=csv seed=2 >> prom-rel.log
python src/train.py experiment=syntactic/prominence/prominence_regression_relative_gpt2_npvp logger=csv seed=2 >> prom-rel.log


python src/train.py experiment=syntactic/prominence/prominence_regression_relative_gpt2 logger=csv seed=3 >> prom-rel.log
python src/train.py experiment=syntactic/prominence/prominence_regression_relative_gpt2_np logger=csv seed=3 >> prom-rel.log
python src/train.py experiment=syntactic/prominence/prominence_regression_relative_gpt2_npvp logger=csv seed=3 >> prom-rel.log