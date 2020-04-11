#!/bin/bash
#SBATCH --partition=physical
#SBATCH --time=0-00:10:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --output=1n1c.out

module load Python/3.4.3-goolf-2015a
time mpirun -np 1 python assignment1.py
