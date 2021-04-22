#!/bin/bash
#SBATCH --mail-user=kathi.woitas@ub.unibe.ch
#SBATCH --mail-type=all
#SBATCH --job-name="e-rara"
#SBATCH --time=02-00:00:00
#SBATCH --mem-per-cpu=20G
#SBATCH --tmp=10GB
#SBATCH --partition=all
#SBATCH --nodes=5
#SBATCH --ntasks-per-node=1

module load Anaconda3
cd $HOME/e-rara/data
python $HOME/e-rara/code/flair_ner_multi_fast.py
