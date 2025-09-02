#!/usr/bin/bash
#SBATCH --job-name=jdfft
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err
#SBATCH --partition gpu
#SBATCH --constraint=h100
#SBATCH --exclusive
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-gpu=80G
#SBATCH --time=00-00:10

module purge
module load python/3.13

export PYTHONUNBUFFERED=TRUE

source $VENVDIR/jdfft-venv/bin/activate

echo "resolution: $1"

if [ "$2" == "double" ]; then
    echo "Using double precision"
    if [ "$SLURM_GPUS_PER_NODE" -gt 1 ] || [ "$SLURM_JOB_NUM_NODES" -gt 1 ]; then
      srun --cpu-bind=cores python jdfft.py --res $1 --double --distributed
    else
        srun --cpu-bind=cores python jdfft.py --res $1 --double
    fi
else
    echo "Using single precision"
    if [ "$SLURM_GPUS_PER_NODE" -gt 1 ] || [ "$SLURM_JOB_NUM_NODES" -gt 1 ]; then
      srun --cpu-bind=cores python jdfft.py --res $1 --distributed
    else
        srun --cpu-bind=cores python jdfft.py --res $1
    fi
fi
