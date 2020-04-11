#!/bin/bash
#SBATCH --partition=physical
#SBATCH --time=0-00:10:00
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=1
#SBATCH --output=2n8c.out

module load Python/3.4.3-goolf-2015a

time mpirun -np 8 python assignment1.py
