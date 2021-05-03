#PBS -l walltime=48:00:00
#PBS -l select=1:ncpus=8:mem=10gb
#PBS -N Post-Processing-Round2

module load anaconda3/personal
source activate personalpy3
date

python /rds/general/user/cfn18/home/Instantons/L96-EBM-Instanton/Brute-Force/Post-Process/postProcess.py 
