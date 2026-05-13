#!/bin/bash

#SBATCH --job-name="dur2parse"
#SBATCH --output="%x.o%j"
#SBATCH --time=120:00:00
#SBATCH --gres=gpu
#SBATCH --partition=spot
#SBATCH --mem=0
#SBATCH --mail-user=jm3743@georgetown.edu
#SBATCH --mail-type=END,FAIL

source env.sh
python -m constituency.wp2parse --use_duration --batch_size 32 --dyck >> libri-dur2dyck.log
