#!/bin/bash
module load anaconda3/3.11
module load cuda/12.5
nvcc --version
source ../prosody/bin/activate
python3 -V
