#!/bin/sh
# Grid Engine options (lines prefixed with #$)
#$ -N retrofit_gpt2       
#$ -cwd         
# -l h_rt=96:00:00 
#$ -l h_vmem=48G
#$ -q gpu 
#$ -pe gpu-a100 1
#$ -R y
# These options are:
# job name: -N
# use the current working directory: -cwd
# runtime limit of 5 minutes: -l h_rt
# memory limit of 1 Gbyte: -l h_vmem

# Initialise the environment modules
. /etc/profile.d/modules.sh

module unload python cuda anaconda
module load anaconda/2024.02 cuda/12.1.1
conda activate /home/s2497456/mnt/run
export TRANSFORMERS_CACHE=/home/s2497456/mnt/cache

# Run the program
export WANDB_DISABLED=true # restrict the usage of wandb
python ./gpt2.py --report_to none
