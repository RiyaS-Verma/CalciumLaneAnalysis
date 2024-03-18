#!/bin/bash
export REPO_PATH=/project/mlmccain_913/riya/calcium_analysis/snakemake
if [ ! -e “slurm_logs” ]
then
  mkdir “slurm_logs”
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
--cluster ‘sbatch --ntasks={cluster.tasks} --cpus-per-task={cluster.cores} --mem={cluster.mem} --time={cluster.time} --account={cluster.acc} --output {cluster.logout} --error {cluster.logerror} module load conda’ \
$@