#!/bin/bash
##
## GPU submission script for PBS on CR2 
## -------------------------------------------
##
## Follow the 6 steps below to configure your job
##
## STEP 1:
##
## Enter a job name after the -N on the line below:
##
#PBS -N cuda_results
##
## STEP 2:
##
## Select the number of cpus/cores and GPUs required by modifying the #PBS -l select line below
##
## The Maximum value for ncpus is 8 and mpiprocs MUST be the same value as ncpus.
## The Maximum value for ngpus is 1 
## e.g.  1 GPU and 8 CPUs : select=1:ncpus=8:mpiprocs=8;ngpus=1
##
#PBS -l select=1:ncpus=1:mpiprocs=1:ngpus=1:mem=32g
##
## STEP 3:
##
## The queue for GPU jobs is defined in the #PBS -q line below
##
#PBS -q gpu_T4
#
## The default walltime in the gpu_A100 queue is one day(24 hours)
## The maximum walltime in the gpu_A100 queue is five days(120 hours)
## In order to increase the walltime modify the #PBS -l walltime line below
## and remove one of the leading # characters
##
##PBS -l walltime=24:00:00
##
## STEP 4:
##
## Replace the hpc@cranfield.ac.uk email address
## with your Cranfield email address on the #PBS -M line below:
## Your email address is NOT your username
##
#PBS -m abe
#PBS -M julien.godfroy.183@cranfield.ac.uk
##
## ====================================
## DO NOT CHANGE THE LINES BETWEEN HERE
## ====================================
#PBS -j oe
#PBS -v "CUDA_VISIBLE_DEVICES="
#PBS -W sandbox=PRIVATE
#PBS -k n
ln -s $PWD $PBS_O_WORKDIR/$PBS_JOBID
## Allocated gpu(s)
echo CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES
## Change to working directory
cd $PBS_O_WORKDIR
## Calculate number of CPUs and GPUs
export cpus=`cat $PBS_NODEFILE | wc -l`
export gpus=`echo $CUDA_VISIBLE_DEVICES|awk -F"," '{print NF}'`
## ========
## AND HERE
## ========
##
## STEP 5:
##
## Load the production USE
module use /apps/modules/all
##  Load the default application environment
##
module load GCC/11.3.0 CUDA/11.7.0 CMake/3.24.3-GCCcore-11.3.0

module load CUDA/12.0.0
##
## STEP 6:
##
## Run gpu application
##
## Put correct parameters and cuda application in the line below:
##
./CUDA_matmult

## Tidy up the log directory
## DO NOT CHANGE THE LINE BELOW
## ============================
rm $PBS_O_WORKDIR/$PBS_JOBID
#
