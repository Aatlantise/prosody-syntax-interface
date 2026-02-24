#!/bin/bash

#SBATCH --job-name="run-mfa"
#SBATCH --output="%x.o%j"
#SBATCH --time=72:00:00
#SBATCH --gres=gpu
#SBATCH --mem=0
#SBATCH --mail-user=jm3743@georgetown.edu
#SBATCH --mail-type=END,FAIL

source $(conda info --base)/etc/profile.d/conda.sh
conda activate aligner
python -m constituency.candor.prep_for_mfa_candor >> candor-proc.log
bash constituency/candor/mfa.sh >> candor-proc.log
python -m constituency.candor.merge_mfa_durations_candor >> candor-proc.log
