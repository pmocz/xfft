#!/bin/bash

# sbatch --time=0-00:10 sbatch_rusty.sh 1024
sbatch --time=0-00:10 --nodes=1 --ntasks-per-node=4 --gpus-per-node=4 --cpus-per-task=1  sbatch_rusty.sh 512

#sbatch --time=0-00:10 sbatch_rusty.sh 64
#sbatch --time=0-00:10 sbatch_rusty.sh 128
#sbatch --time=0-00:10 sbatch_rusty.sh 256
#sbatch --time=0-00:10 sbatch_rusty.sh 512
#sbatch --time=0-00:10 sbatch_rusty.sh 1024

#sbatch --time=0-00:10 sbatch_rusty.sh 64 double
#sbatch --time=0-00:10 sbatch_rusty.sh 128 double
#sbatch --time=0-00:10 sbatch_rusty.sh 256 double
#sbatch --time=0-00:10 sbatch_rusty.sh 512 double


# XXX
# sbatch --time=0-00:10 --nodes=1 --ntasks-per-node=4 --gpus-per-node=4 --cpus-per-task=1  sbatch_rusty.sh 512
# sbatch --time=0-20:00 --nodes=2 --ntasks-per-node=4 --gpus-per-node=4 --cpus-per-task=1 sbatch_rusty.sh 512
# sbatch --time=0-20:00 --nodes=2 --ntasks-per-node=4 --gpus-per-node=4 --cpus-per-task=1 sbatch_rusty.sh 1024
