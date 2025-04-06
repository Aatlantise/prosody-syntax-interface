#!/bin/bash

#SBATCH --job-name="pause"
#SBATCH --output="%x.o%j"
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu
#SBATCH --time=12:00:00
#SBATCH --mem=0
#SBATCH --mail-user=jm3743@georgetown.edu
#SBATCH --mail-type=END,FAIL

source env.sh

python src/train.py experiment=syntactic/pause/pause_regression_after_gpt2 seed=1 logger=csv >> pause.log
python src/train.py experiment=syntactic/pause/pause_regression_after_gpt2_np seed=1 logger=csv >> pause.log
python src/train.py experiment=syntactic/pause/pause_regression_after_gpt2_npvp seed=1 logger=csv >> pause.log

python src/train.py experiment=syntactic/pause/pause_regression_after_gpt2 seed=2 logger=csv >> pause.log
python src/train.py experiment=syntactic/pause/pause_regression_after_gpt2_np seed=2 logger=csv >> pause.log
python src/train.py experiment=syntactic/pause/pause_regression_after_gpt2_npqvp seed=2 logger=csv >> pause.log

python src/train.py experiment=syntactic/pause/pause_regression_after_gpt2 seed=3 logger=csv >> pause.log
python src/train.py experiment=syntactic/pause/pause_regression_after_gpt2_np seed=3 logger=csv >> pause.log
python src/train.py experiment=syntactic/pause/pause_regression_after_gpt2_npvp seed=3 logger=csv >> pause.log