#!/bin/bash
#SBATCH --job-name=tolokers
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=2
#SBATCH --cpus-per-task=16
#SBATCH --time=20-12
#SBATCH --constraint=type_h
#SBATCH --output=tolokers_results.txt
#SBATCH --error=tolokers_errors.txt
#SBATCH --partition=rocky


module purge && module load Python
conda activate nikita_neurips

# Run the same launcher script once per node
srun --label --kill-on-bad-exit=1 bash ./run_all_experiments_dual_gpu_alex.sh