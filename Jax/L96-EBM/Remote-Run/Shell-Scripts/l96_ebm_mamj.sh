#PBS -l walltime=24:00:00
#PBS -l select=1:ncpus=48:mem=124gb
#PBS -N MAMJ-dt0_1-steps300

module load anaconda3/personal
source activate personalpy3
date

python $PBS_O_WORKDIR/mamj_L96ebm.py
