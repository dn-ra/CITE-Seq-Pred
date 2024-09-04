#!/bin/bash
#while IFS=',' read -r adt idx dsb_adt cellbender_rna; do 


#dsb_adt='--dsb_adt'
cellbender_rna='--cellbender_rna'
dataset='scvi'
adt='CD25_TotalSeqB'
idx=3
    sbatch <<EOT
#!/bin/bash
#SBATCH --job-name=GCN_various_denoising_scvi
#SBATCH --output=./GCN_slurm_logs/%j.out
#SBATCH --error=./GCN_slurm_logs/%j.err
#SBATCH --time=01:00:00
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
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/danrawlinson/.conda/envs/scvi/lib/:/home/danrawlinson/.conda/envs/scvi/lib/python3.9/site-packages/tensorrt/
python GCN_various_denoising.py --dataset $dataset --adt $adt --kfold $idx $dsb_adt $cellbender_rna \
                                --epochs 10 --batch_size 100 --graph_strategy "together"


EOT


#done < combinations.txt

