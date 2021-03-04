# Make directory to copy model in to, submit job from there
NOW=$(date +"%Y-%m-%d-%T")
run_directory=$EPHEMERAL/L96-EBM-Stochastic/$NOW
mkdir $run_directory
cp -r $HOME/Instantons/L96-EBM-Instanton/Brute-Force/Remote-Run $run_directory
cd $run_directory/Remote-Run
cp $run_directory/Remote-Run/Shell-Scripts/l96_ebm_sto.sh .
qsub l96_ebm_sto.sh

