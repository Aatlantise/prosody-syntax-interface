#!/bin/bash

#SBATCH --job-name="spi-baseline"
#SBATCH --output="%x.o%j"
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu
#SBATCH --time=48:00:00
#SBATCH --mem=0
#SBATCH --mail-user=jm3743@georgetown.edu
#SBATCH --mail-type=END,FAIL

source env.sh
python reverse/run.py --seed 1 >> baseline-1.log
python reverse/run.py --seed 2 >> baseline-2.log
python reverse/run.py --seed 3 >> baseline-3.log
