#!/bin/bash

#SBATCH --job-name="spi-duration"
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
python reverse/run.py --use_duration_info --seed 1 >> duration-1.log
python reverse/run.py --use_duration_info --seed 2 >> duration-2.log
python reverse/run.py --use_duration_info --seed 3 >> duration-3.log