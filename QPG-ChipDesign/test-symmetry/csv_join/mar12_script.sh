#!/bin/bash
#
#SBATCH --nodes=1
#SBATCH --open-mode=append
#SBATCH --output=/home/tareq/public_html/logs/%A_%a.out
#SBATCH --error=/home/tareq/public_html/logs/%A_%a.err
#S BATCH --mail-type=ALL
#S BATCH --mail-user=tareq@mit.edu

module load engaging/python/3.6.0

python3 ~/QPG-ChipDesign/test-symmetry/csv_join/mar12_join.py