##########################################
## TO DO PRE SUBMITTING:
## - Set Parameters
## - Set save_directory 
## - Update PBS Name + settings in other shell script
##########################################

##########################################
## Imports
##########################################

# MAM Code Imports
import sys
sys.path.append('/rds/general/user/cfn18/home/Instantons/L96-EBM-Instanton')
from fw_action import *
from mam_jax import *

# L96 EBM imports
from utilities import b, diff_inv, L96EBMMO

# Standard Imports
import os
import numpy.random as rm

##########################################
## Save Details
##########################################

# Results saved after each blocks
# Max run len will be: blocks * block_len
blocks = 100
block_len = 50

# Where We Save Output
save_directory = f'/rds/general/user/cfn18/home/Instantons/L96-EBM-Instanton/Jax/L96-EBM/Data/dt0_1steps300'

if not os.path.exists(save_directory):
    os.makedirs(save_directory)

##########################################
## Setting Parameters
##########################################

# L96 EBM Parameters
K = 50
S = 12.
a0 = 0.5
a1 = 0.4
sigma = 1/180**4
F = 8.
Tref = 270.
delT = 60.
alpha = 2.
beta = 1.

# Noise Parameters
eps = 1.e-3
delta = 0.1

p = [K, S, a0, a1, sigma, F, Tref, delT, alpha, beta, eps, delta]

# Time
steps = 300
dt = 0.1
time = np.arange(0, dt * (steps + 1), dt) #+1 for 0 step

##########################################
## Making I.C. for MAM
##########################################

initial_x = 2.8 + rm.random(K)
initial_temp = 260
initial_point = np.append(initial_x, initial_temp)

final_x = 2.8 + rm.random(K)
final_temp = 295
final_point = np.append(final_x, final_temp)

inst_ic = np.linspace(initial_point, final_point, steps + 1)
path_shape = inst_ic.shape

##########################################
## Setting Up MAM Objects
##########################################

# Object for calculating the FW Action and it's jacobian 
jfw = JFW(b, diff_inv) 

# Minimisation object
mamjax = MamJax(jfw, inst_ic, time, p)

# Observer object
observer = L96EBMMO(mamjax, save_directory)

##########################################
## Running and Saving in Blocks
##########################################

# Runnning MAM
opt={'maxiter': block_len, 'maxfun': block_len}

print('\n*** Starting MAM *** \n')

for i in range(blocks):
    print(f'\nRunning block {i}\n')
    mamjax.run(opt)
    observer.snapshot() # Result saved after each block
    
    # Check if you've converged
    if (mamjax.res.success is True):
        print('Success, quitting MAM and saving result')
        break
