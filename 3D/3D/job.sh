#!/bin/bash
#SBATCH -J jacobi                       # SLURM_JOB_NAME      ==> jobname
#SBATCH -N 10                            # SLURM_JOB_NUM_NODES ==> @course [1-4]
#SBATCH --reservation=sohpc_vsc3_mem_0128
#SBATCH --tasks-per-node=16             # SLURM_NTASKS_PER_NODE [default: 16/32]
#SBATCH --export=NONE                   # do not inherit the submission env
###SBATCH --get-user-env                  # run system profile         [?maybe?]
#SBATCH --time=00:10:00                 # time limit         ==> @course [1 min]
#SBATCH --partition=mem_0128
#SBATCH --qos=normal_0128

module purge                            # always start with a clean environment
module load intel/18 intel-mpi/2018     # load modules needed (? mpi names)

echo
echo '=== JACOBI - RUNNING ON: ================================================'
echo 
echo '=== SLURM_CLUSTER_NAME  = '$SLURM_CLUSTER_NAME
echo '=== SLURM_JOB_PARTITION = '$SLURM_JOB_PARTITION
echo '=== SLURM_JOB_NODELIST  = '$SLURM_JOB_NODELIST
echo
echo '=== SLURM_JOB_NUM_NODES = '$SLURM_JOB_NUM_NODES
echo '=== MPI_PROCESSES       = '$MPI_PROCESSES


export I_MPI_PIN_PROCESSOR_LIST=0-15

mpirun -n 160 ./jacobi.exe < input
