#!/bin/bash

#SBATCH --job-name="pause2parse"
#SBATCH --output="%x.o%j"
#SBATCH --time=72:00:00
#SBATCH --gres=gpu
#SBATCH --mem=0
#SBATCH --mail-user=jm3743@georgetown.edu
#SBATCH --mail-type=END,FAIL

source env.sh
python -m constituency.wp2parse --use_pause >> pause2parse.log