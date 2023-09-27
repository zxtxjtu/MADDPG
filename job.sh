#!/bin/bash
#SBATCH -J main.py
#SBATCH -o test.out
#SBATCH -N 1
#SBATCH -n 32
#SBATCH --ntasks-per-node=32
#SBATCH -t 1:00:00
#SBATCH --gres=gpu:2
