# Integrator Code Imports
import sys
sys.path.append('/rds/general/user/cfn18/home/Instantons/L96-EBM-Instanton')

from l96EBM import * # IMPORT TEST

# Standard Package imports
import numpy as np
import numpy.random as rm
import os
import sys


##########################################
## Where Output is Written
##########################################

save_directory = '/rds/general/user/cfn18/ephemeral/L96-EBM-Deterministic/Ensemble-Run20'

##########################################
## Setting Parameters
##########################################

# L96 Parameters
K = 20
S = 12
a0 = 0.5
a1 = 0.4
sigma = 1/180**4
F = 8
Tref = 270
delT = 60
alpha = 2
beta = 1
p = [K, S, a0, a1, sigma, F, Tref, delT, alpha, beta]

# Number of observations
obs_freq = 1
num_obs = 100000 # How many observations for each trajectory

##########################################
## Choosing Spread of Intial Ts
##########################################

T0s = []
for temp in [258, 275.5, 293]: # one run for each attractor
    T0s.append(temp + rm.normal(scale=2, size=1))
    
##########################################
## Running Model
##########################################

for T in T0s:
    runner = Integrator(p=p, T_init=T)
    looker = TrajectoryObserver(runner, name=f'T_{T.item():.2f}'.replace('.', '_'))
    make_observations(runner, looker, num_obs, obs_freq)
    looker.dump(save_directory)