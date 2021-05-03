#PBS -l walltime=24:00:00
#PBS -l select=1:ncpus=8:mem=10gb
#PBS -N k50_eps0_1
#PBS -J 1-1000

module load anaconda3/personal
source activate diffeqpy_env
date

python $PBS_O_WORKDIR/l96EbmSto.py $PBS_ARRAY_INDEX
