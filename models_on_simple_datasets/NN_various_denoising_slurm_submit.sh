#!/bin/bash
while IFS=',' read -r adt idx dsb_adt cellbender_rna; do 


# dsb_adt='--dsb_adt'
# cellbender_rna='--cellbender_rna'
dataset='nextgem'
# adt='PD-1_TotalSeqB'
# idx=3
    sbatch <<EOT
#!/bin/bash
#SBATCH --job-name=NN_various_denoising_scvi
#SBATCH --output=./NN_slurm_logs/%j.out
#SBATCH --error=./NN_slurm_logs/%j.err
#SBATCH --time=00:10:00
#SBATCH --partition=mig-gpu,gpu-a100-short
#SBATCH --gres=gpu:1
#SBATCH --account=punim0613
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=10GB



#print the job information
echo "Job ID: \$SLURM_JOB_ID"
echo "Job Name: \$SLURM_JOB_NAME"
echo "args:" $dataset $adt $idx $dsb_adt $cellbender_rna

#run the python script with scvi conda interpreter
source ~/.bashrc
export WANDB_CACHE_DIR='/data/gpfs/projects/punim1597/Projects/CITE-seq/models_on_simple_datasets/.wandb_cache'
conda activate scvi
python NN_various_denoising.py --dataset $dataset --adt $adt --kfold $idx $dsb_adt $cellbender_rna


EOT


done < combinations.txt

