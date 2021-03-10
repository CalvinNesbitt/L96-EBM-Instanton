#PBS -l walltime=00:30:00
#PBS -l select=1:ncpus=8:mem=10gb
#PBS -N Post-Processing

module load anaconda3/personal
source activate personalpy3
date

python /rds/general/user/cfn18/home/Instantons/L96-EBM-Instanton/Brute-Force/Post-Process/postProcess.py 
