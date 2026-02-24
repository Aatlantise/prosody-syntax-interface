#!/bin/bash

#SBATCH --job-name="stanza"
#SBATCH --output="%x.o%j"
#SBATCH --time=24:00:00
#SBATCH --gres=gpu
#SBATCH --mem=0
#SBATCH --mail-user=jm3743@georgetown.edu
#SBATCH --mail-type=END,FAIL

source env.sh
python -m constituency.util >> stanza.log
