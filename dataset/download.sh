#!/bin/sh
# Grid Engine options (lines prefixed with #$)
#$ -N download       
#$ -cwd         
#$ -l h_rt=08:00:00 
#$ -l h_vmem=32G
# These options are:
# job name: -N
# use the current working directory: -cwd
# runtime limit of 5 minutes: -l h_rt
# memory limit of 1 Gbyte: -l h_vmem

# Initialise the environment modules
. /etc/profile.d/modules.sh
export HF_HOME=/home/s2497456/mnt/cache

module unload python anaconda
module load anaconda/2024.02 
conda activate /home/s2497456/mnt/workdir/run
#conda config --set envs_dirs /home/s2497456/mnt/cache

python clean.py

# Run the program
python ./download.py
