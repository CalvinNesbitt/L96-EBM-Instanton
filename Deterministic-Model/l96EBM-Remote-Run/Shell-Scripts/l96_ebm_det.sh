#PBS -l walltime=24:00:00
#PBS -l select=1:ncpus=8:mem=10gb
#PBS -N L96-Ensemble40
#PBS -J 1-50

module load anaconda3/personal
source activate personalpy3
date

python $PBS_O_WORKDIR/l96EbmEnsemble.py $PBS_ARRAY_INDEX
