"""
Functions for post processing the output of an L96 EBM stochastic integration.
"""

import xarray as xr
import numpy as np
import os
import pickle
from tqdm import tqdm

################################################
## Input/Output
################################################

def pickler(l, s):
    "Pickle list l at location s"
    with open(s + '.pickle', 'wb') as output_file:
        pickle.dump(l, output_file)
        
def unpickler(s):
    "Unpickle s.pickle."
    with open(s + '.pickle', "rb") as input_file:
        return pickle.load(input_file)
    
def close_list(run):
    for ds in run:
        ds.close()
    print('Closed Runs')
    
def list_runs(d, s=slice(0, -1, 1)):
    """
    Get list of all directories containing *.nc files in array jobs run.
    
    d, param, string
        Name of array jobs directory.
    s, param, slice
        Slice to index the array jobs we filter through.
    
    run_list, output, list
        List of directory names.
        Each directory will contatin a load of .nc files.
    """
    run_list = [] 
    runs = os.listdir(d) # list of array job runs
    runs.sort(key=float)
    for file in runs[s]:
        folder = d + '/' + file +'/'
        for Tfile in os.listdir(folder):
            subfolder = folder + Tfile
            run_list.append(subfolder)
    return run_list


################################################
## Sorting Runs
################################################

def re_test(ds, thres=(270, 280)):
    """
    Tests if a L96 EBM dataset contains a transition.
    Returns boolean.
    
    ds, xr.dataset, param
        Dataset containing the L96 EBM run.
        
    thres, tuple, param
        Thresholds for transition test.
        Form is (cold, hot).
        Transition defined by (x<cold & x>hot) in the timeseries.
        
    transition, boolean, output
        Did we have a transition?
    """
    cold = ds.T.values<thres[0]
    hot = ds.T.values>thres[1]
    transition = np.any(cold) & np.any(hot)
    return transition

def sort_runs(run_list, thres=(270, 280), prog=False):
    """
    Sorts an array jobs directory in to transitions, no transitions and corrupt runs.
    
    run_list, param, list
        List of directory names.
        Can be produced by list_runs() for example.
        Each directory should contatin a load of .nc files.
    thres, option, tuple
        Thresholds used to test for transitions (see re_test).
    prog, option, boolean
        Do you want a progress bar?
        
    [re_list, boring_list, corrupt_list], output, list
        re_list is are directories containing a transition.  
    """

    re_list = []
    boring_list = []
    corrupted_list = []
    
    if (prog):
        print(f'{len(run_list)} runs to sort')
        
    for run in tqdm(run_list, disable=(not prog)):
        corrupted = False
        try:
            # Open and check if it contains a transition
            with xr.open_mfdataset(run + '/*.nc', combine='by_coords', concat_dim='time') as ds:
                rare = re_test(ds, thres)
        except:
            print(f'Error opening {run}')
            corrupted = True
            
        # Add to the correct list
        if corrupted:
            corrupted_list.append(run)
        elif ((not corrupted) & rare):
            re_list.append(run)
        elif (not corrupted):
            boring_list.append(run)
            
    return [re_list, boring_list, corrupted_list]

def sort_n_save(run_list, save_dir, prog=False):
    """
    Sorts a list of run directories in to rare ones, boring ones
    and corrupt ones. Pickles sorted lists at save_dir.
    
    run_list, param, list
        List of run directories.
    save_dir, param, string
        Where to save output.
    prog, option, boolean
        Want a progress bar?
    """
    
    # Lists contain rare, boring and corrupt run directories
    r_list = []
    b_list = []
    c_list = []
    lists = [r_list, b_list, c_list]
    list_names = ['rare_list', 'boring_list', 'corrupt_list']

    # Break run list in to chunks of ~5
    chunk_size = int(len(run_list)/5)

    for chunk in tqdm(np.array_split(np.array(run_list), chunk_size), disable =(not prog)):

        rare, boring, corr = sort_runs(chunk)
        r_list += rare
        b_list += boring
        c_list += corr

        # Save output for each chunk
        for l, name in zip(lists, list_names):
            pickler(l, save_dir + '/' + name)
            
    return
        