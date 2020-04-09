#!/bin/bash
#SBATCH --partition=physical
#SBATCH --time=0-00:05:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --output=1n8c.out

module load Python/3.4.3-goolf-2015a

time mpirun -np 8 python assignment1.py
