#!/bin/bash
export REPO_PATH=/project/mlmccain_913/riya/calcium_analysis/snakemake
export OPENBLAS_NUM_THREADS=1 #cluter user openbladnumthreads has to be 1
export JAVA_OPTS="-Xmx32g -XX:ParallelGCThreads=16" #java parameters for bfconvert

if [ ! -e "slurm_logs" ]
then
  mkdir "slurm_logs"
fi

snakemake \
--snakefile $REPO_PATH/Snakefile \
--configfile $REPO_PATH/config.yaml \
--printshellcmds \
--keep-going \
--rerun-incomplete \
--cluster-config $REPO_PATH/cluster.yaml \
--cores 50 \
--jobs 30 \
$@
