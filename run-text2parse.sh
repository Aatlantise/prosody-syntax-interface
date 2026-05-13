#!/bin/bash

#SBATCH --job-name="cd-autoreg"
#SBATCH --gpus=h100:1
#SBATCH --mem=64G
#SBATCH --output="%x.o%j"
#SBATCH --time=144:00:00
#SBATCH --account=def-annielee
#SBATCH --cpus-per-task=4
#SBATCH --mail-user=jm3743@georgetown.edu
#SBATCH --mail-type=END,FAIL

source env.sh
python -m constituency.wp2parse --use_text --dyck >> libri-text2dyck.log
