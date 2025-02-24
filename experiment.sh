#!/bin/bash

#SBATCH --job-name="syntax-prosody-interface"
#SBATCH --output="%x.o%j"
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --mem=4G
#SBATCH --mail-user=jm3743@georgetown.edu
#SBATCH --mail-type=END,FAIL

python src/train.py experiment=emnlp/finetuning/duration_regression_syll_gpt2 logger=csv >> syll.log
python src/train.py experiment=emnlp/finetuning/duration_regression_syll_gpt2_np logger=csv >> syll.log
python src/train.py experiment=emnlp/finetuning/duration_regression_syll_gpt2_npvp logger=csv >> syll.log
python src/train.py experiment=emnlp/finetuning/pause_regression_after_gpt2 logger=csv >> pause.log
python src/train.py experiment=emnlp/finetuning/pause_regression_after_gpt2_np logger=csv >> pause.log
python src/train.py experiment=emnlp/finetuning/pause_regression_after_gpt2__npvp logger=csv >> pause.log
