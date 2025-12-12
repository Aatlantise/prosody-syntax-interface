#!/bin/bash

#SBATCH --job-name="t5-autoreg"
#SBATCH --output="%x.o%j"
#SBATCH --time=36:00:00
#SBATCH --gres=gpu
#SBATCH --mem=0
#SBATCH --mail-user=jm3743@georgetown.edu
#SBATCH --mail-type=END,FAIL

source env.sh
python -m constituency.wp2parse >> t5_autoreg.log