#!/bin/bash

#SBATCH --job-name="spi-pause"
#SBATCH --output="%x.o%j"
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu
#SBATCH --partition=spot
#SBATCH --time=48:00:00
#SBATCH --mem=0
#SBATCH --mail-user=jm3743@georgetown.edu
#SBATCH --mail-type=END,FAIL

source env.sh
python reverse/run.py --use_pause_info --seed 1 >> pause-1.log
python reverse/run.py --use_pause_info --seed 2 >> pause-2.log
python reverse/run.py --use_pause_info --seed 3 >> pause-3.log