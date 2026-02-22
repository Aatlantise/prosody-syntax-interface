#!/bin/bash

#SBATCH --job-name="candor-proc"
#SBATCH --output="%x.o%j"
#SBATCH --time=48:00:00
#SBATCH --gres=gpu
#SBATCH --mem=0
#SBATCH --mail-user=jm3743@georgetown.edu
#SBATCH --mail-type=END,FAIL

source env.sh
module unload anaconda3/3.11
module load anaconda3/3.13
python -m constituency.candor.get_surprisals_candor >> candor-proc.log
python -m constituency.candor.process_av_features_candor >> candor-proc.log
deactivate
conda activate aligner
python -m constituency.candor.prep_for_mfa_candor >> candor-proc.log
python -m constituency.candor.run_mfa_candor >> candor-proc.log
