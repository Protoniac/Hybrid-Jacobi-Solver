#!/bin/bash
#SBATCH -J jacobi                       # SLURM_JOB_NAME      ==> jobname
#SBATCH -N 12                            # SLURM_JOB_NUM_NODES ==> @course [1-4]
#SBATCH --reservation=sohpc_vsc3_vsc3plus_0256
#SBATCH --tasks-per-node=20             # SLURM_NTASKS_PER_NODE [default: 16/32]
#SBATCH --export=NONE                   # do not inherit the submission env
###SBATCH --get-user-env                  # run system profile         [?maybe?]
#SBATCH --time=00:10:00                 # time limit         ==> @course [1 min]
#SBATCH --partition=vsc3plus_0256
#SBATCH --qos=vsc3plus_0256

module purge                            # always start with a clean environment
module load intel/18 intel-mpi/2018     # load modules needed (? mpi names)

# Set (max) number of MPI processes
  export MPI_PROCESSES=$SLURM_NTASKS    # number of MPI processes

echo
echo '=== JACOBI - RUNNING ON: ================================================'
echo 
echo '=== SLURM_CLUSTER_NAME  = '$SLURM_CLUSTER_NAME
echo '=== SLURM_JOB_PARTITION = '$SLURM_JOB_PARTITION
echo '=== SLURM_JOB_NODELIST  = '$SLURM_JOB_NODELIST
echo
echo '=== SLURM_JOB_NUM_NODES = '$SLURM_JOB_NUM_NODES
echo '=== MPI_PROCESSES       = '$MPI_PROCESSES

export I_MPI_PIN_PROCESSOR_LIST=0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19
mpirun -n 240 ./jacobi.exe < input
