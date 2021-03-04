# Make directory to copy model in to, submit job from there
NOW=$(date +"%Y-%m-%d-%T")
run_directory=$EPHEMERAL/L96-EBM-Deterministic/$NOW
mkdir $run_directory
cp -r $HOME/Instantons/L96-EBM-Instanton/Deterministic-Model/l96EBM-Remote-Run $run_directory
cd $run_directory/l96EBM-Remote-Run/
cp $run_directory/l96EBM-Remote-Run/Shell-Scripts/l96_ebm_det.sh .
qsub l96_ebm_det.sh