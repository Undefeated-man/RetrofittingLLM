#!/bin/sh
# Grid Engine options (lines prefixed with #$)
#$ -N hello       
#$ -cwd         
#$ -l h_rt=00:05:00 
#$ -l h_vmem=5G
#$ -q gpu 
#$ -pe gpu-a100 1
# These options are:
# job name: -N
# use the current working directory: -cwd
# runtime limit of 5 minutes: -l h_rt
# memory limit of 1 Gbyte: -l h_vmem

# Initialise the environment modules
. /etc/profile.d/modules.sh

module unload python cuda anaconda
module load anaconda/2024.02 cuda/12.1.1
conda activate /home/s2497456/mnt/workdir/run

nvidia-smi

# Run the program
python ./hello.py
