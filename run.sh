#!/bin/bash
#SBATCH -N 1
#SBATCH -c 1
#SBATCH --gres=gpu
#SBATCH -p ug-gpu-small
#SBATCH --qos=short
#SBATCH -t 01-00:00:00
#SBATCH --job-name=pix2pixLaurie
#SBATCH --mem=32G
#SBATCH --mail-user laurence.ho@durham.ac.uk
#SBATCH --mail-type=ALL

source ~/pix2pixenv/bin/activate
module load cuda/10.1-cudnn7.6


python train.py --dataroot ./datasets/sketchy --name sketchy_pix2pix --model pix2pix --direction BtoA --display_id=0

python train.py --dataroot ./datasets/sketchy --name sketchy_pix2pix --model pix2pix --direction BtoA --display_id=0