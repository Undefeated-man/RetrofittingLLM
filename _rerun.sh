#!/bin/sh

python3 clean.py
git pull
qsub run.sh
