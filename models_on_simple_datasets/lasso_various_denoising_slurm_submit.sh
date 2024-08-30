#!/bin/bash
#while IFS=',' read -r adt idx dsb_adt cellbender_rna; do 
dataset='nextgem'
adt='CD127_TotalSeqB'
idx=5
dsb_adt='--dsb_adt'
cellbender_rna='--cellbender_rna'

    sbatch <<EOT
#!/bin/bash
#SBATCH --job-name=lasso_various_denoising_nextgem
#SBATCH --output=./slurm_logs/%j.out
#SBATCH --error=./slurm_logs/%j.err
#SBATCH --time=01:00:00
#SBATCH --partition=cascade,sapphire
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=5GB


#print the job information
echo "Job ID: \$SLURM_JOB_ID"
echo "Job Name: \$SLURM_JOB_NAME"
echo "args:" $dataset $adt $idx $dsb_adt $cellbender_rna

#run the python script with scvi conda interpreter
source ~/.bashrc
conda activate scvi
export WANDB_CACHE_DIR='/data/gpfs/projects/punim1597/Projects/CITE-seq/models_on_simple_datasets/.wandb_cache'
python lasso_various_denoising.py --dataset $dataset --adt $adt --kfold $idx $dsb_adt $cellbender_rna

EOT

#done < combinations.txt
