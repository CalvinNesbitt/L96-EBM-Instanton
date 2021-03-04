##########################################
## TO DO PRE SUBMITTING:
## - Set Parameters
## - Set save_directory 
## - Update PBS Name + settings in other shell script
##########################################


##########################################
## Integrator Code Imports
##########################################
import sys
sys.path.append('/rds/general/user/cfn18/home/Instantons/L96-EBM-Instanton/Brute-Force')

from l96ESUtility import * # IMPORT TEST

#diffeqpy requires Julia
from julia.api import Julia
jl = Julia(compiled_modules=False)
from diffeqpy import de
import numba

# Standard Package imports
import numpy as np
import numpy.random as rm
import os


##########################################
## Where Output is Written
##########################################

save_directory = f'/rds/general/user/cfn18/ephemeral/L96-EBM-Stochastic/k40_eps0_01/{str(sys.argv[1])}'
os.makedirs(save_directory)

##########################################
## Setting Parameters
##########################################

# L96 EBM Parameters
K = 40
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
eps = 0.01
delta = 0.1

p = np.array([K, S, a0, a1, sigma, F, Tref, delT, alpha, beta, eps, delta])

# Number of observations
block_length = 1000 # Output stored every 0.1 so no. of obs = blocklenth/0.1 * number of blocks
number_of_blocks = 10

##########################################
## Choosing Spread of Intial Ts
##########################################

T0s = []
for temp in [258, 275, 293]: # one run for each attractor
    T0s.append(temp + rm.normal(scale=2, size=1))
    
##########################################
## Problem Definition
##########################################

def f(du, u, p, t): # Drift
    p = [K, S, a0, a1, sigma, F, Tref, delT, alpha, beta, eps, delta]
    X, T = u[:K], u[K:]

    du[:K] = (
        np.roll(X, 1) * (np.roll(X, -1) - np.roll(X, 2)) - 
        X + F * (1 + beta * (T - Tref)/delT )
        )
    
    avg_energy = 0.5 * np.sum(X**2)/K
    du[K:] = (
            S * (1 - a0 + 0.5 * a1 * (np.tanh(T - Tref))) 
            - sigma * T**4 - alpha * (avg_energy/(0.6 * F**(4/3)) - 1)
        )
    
def g(du, u, p, t): # Diffusion
    p = [K, S, a0, a1, sigma, F, Tref, delT, alpha, beta, eps, delta]
    X, T = u[:K], u[K:]
    
    du[:K] = np.sqrt(eps) * delta
    du[K:] = np.sqrt(eps) * (S * (1 - a0 + 0.5 * a1 * (np.tanh(T - Tref))))
    
# Numba for Speed    
numba_f = numba.jit(f)
numba_g = numba.jit(g)
    
##########################################
## Running Model
##########################################

for T in T0s:
    
    # Initial Condition
    u0 = np.append(5 + rm.randn(K), T) 

    # Directory where we save this run
    sd = save_directory + f'/T_{T.item():.6f}'.replace('.', '_') 
    os.mkdir(sd)

    # Setting up diffeqpy solver
    t = 0.
    tspan = (0., block_length)
    prob = de.SDEProblem(numba_f, numba_g, u0, tspan, p)
    looker = SolutionObserver(prob.p)

    for i in range(number_of_blocks):
        print(prob.tspan)

        # Integrate
        sol = de.solve(prob, de.SOSRI(), reltol=1e-3,abstol=1e-3, saveat=0.1)
        looker.look(sol)

        # Save Observations
        looker.dump(sd)

        # Update Step
        t += block_length
        prob = de.remake(prob, u0=sol.u[-1], tspan=(t, t + block_length))
