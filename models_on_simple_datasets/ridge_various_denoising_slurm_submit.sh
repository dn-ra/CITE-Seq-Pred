#!/bin/bash
while IFS=',' read -r adt idx dsb_adt cellbender_rna; do 

#only run for a single adt - make it an easy one 
if [ "$adt" != "CD3_TotalSeqB" ]; then
    echo "adt is not 'CD3'. Skipping."
    continue
fi
 
#adt='CD25_TotalSeqB'


    sbatch <<EOT
#!/bin/bash
#SBATCH --job-name=ridge_various_denoising_scvi
#SBATCH --output=./ridge_slurm_logs/%j.out
#SBATCH --error=./ridge_slurm_logs/%j.err
#SBATCH --time=02:00:00
#SBATCH --partition=cascade,sapphire
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=5GB


#print the job information
echo "Job ID: \$SLURM_JOB_ID"
echo "Job Name: \$SLURM_JOB_NAME"
echo "args:" $adt $idx $dsb_adt $cellbender_rna

#run the python script with scvi conda interpreter
source ~/.bashrc
conda activate scvi
python ridge_various_denoising.py --adt $adt --kfold $idx $dsb_adt $cellbender_rna

EOT

done < combinations.txt
