""" 
Functions used to apply jax asssisted MAM algorithm to the L96 EBM problem.

----------
Contents
----------
- b, L96 EBM Drift function.

- diff_inv, L96 EBM inverse Diffusion function. 

- L96EBMMO, class for observing L96 EBM MAM minimisation. 

"""

##########################################
## Imports
##########################################

import xarray as xr
import pickle
import numpy as np
import jax.numpy as jnp
from jax.ops import index, index_add, index_update

##########################################
## L96 Drift and Diffusion
##########################################


def b(y, p):
    """
    L96 EBM Drift function. 
    Designed for use with MAMJax algorithm. Recall Jax likes pure functions.
    
    Parameters
    ----------
    y, np array 
        Point on instanton path. 
        Expected shape is (dim).
    
    p, list
        L96 Parameters:
        [K, S, a0, a1, sigma, F, Tref, delT, alpha, beta, eps, delta].
        
    return
    ----------
    b, jnp array 
        Drift vector at point on instanton path. 
    """
    
    K, S, a0, a1, sigma, F, Tref, delT, alpha, beta, eps, delta = p
    
    x = y[:-1]
    temp = y[-1:]
    
    # X terms along path
    
    b_x = jnp.roll(x, 1) * (jnp.roll(x, -1) - jnp.roll(x, 2)) - x + F * (1 + beta * (temp - Tref)/delT)
    
    # T terms along path
    avg_energy = 0.5/K * jnp.sum(x**2)
    b_t = S * (1 - a0 + 0.5 * a1 * (jnp.tanh(temp - Tref))) - sigma * temp**4 - alpha * (avg_energy/((0.6 * F**(4/3)) - 1))
    
    return jnp.concatenate((b_x, b_t))

def diff_inv(y, p):
    """
    L96 EBM inverse Diffusion function. 
    Designed for use with MAMJax algorithm. Recall Jax likes pure functions.
    
    Parameters
    ----------
    y, np array 
        Point on instanton path. 
        Expected shape is (dim).
    
    p, list
        L96 Parameters:
        [K, S, a0, a1, sigma, F, Tref, delT, alpha, beta, eps, delta].
        
    return
    ----------
    Dinv, jnp array 
        Inverse of diffusion matrix evaluated at provided point. 
    """
    
    K, S, a0, a1, sigma, F, Tref, delT, alpha, beta, eps, delta = p

    D = delta * np.eye(K+1)
    T = y[-1]
    D = index_update(D, index[-1, -1], 1 - a0 + 0.5 * a1 * jnp.tanh(T - Tref))
    return jnp.linalg.inv(D)

##########################################
## L96 EBM MAM Jax Observer
##########################################

class L96EBMMO:
    """
    Class for observing L96 EBM MAM minimisation. 
    
    - Currently observes: instanton, action value, status and parameters.
    - Initialised with a MAMJAX object and a save location.
    
    Methods
    -----------
    save_parameters()
        Runs when initialised, saves parameters in pickled dictionary.

    snaphot()
        To be called after each bout of minimisation.
        Saves action value, instanton and status in save_loc directory.
    
    Attributes
    -----------
    mj: MamJax
        Object used to rune the MAM algorithm with jax.
        Imagine this would work with a mam object, haven't tried.
    
    save_loc: string 
        Directory where the observations saved.
        
    """
       
    def __init__(self, mj, save_loc):
        """
        Parameters
        ----------
        mj: MamJax
            Object used to rune the MAM algorithm with jax.
            Imagine this would work with a mam object, haven't tried.

        save_loc: string 
            Directory where the observations saved.

        """
        
        self.mj = mj       
        self.save_loc = save_loc
        
        self.save_parameters()
        self.av_list = []
        
    @property    
    def parameters(self):
        "Parameters used for a run. Access by instanton"
        
        # Unpack parameters and put in labelled dictionary
        K, S, a0, a1, sigma, F, Tref, delT, alpha, beta, eps, delta = self.mj.p
        param = {
        'K' : K,
        'S': S,
        'a0': a0,
        'a1': a1,
        'sigma': sigma,
        'F': F,
        'Tref': Tref,
        'delT': delT,
        'alpha': alpha,
        'beta': beta,
        'eps': eps,
        'delta': delta
        }
        return param
        
    def save_parameters(self):
        with open(self.save_loc +'/parameters.pickle','wb') as file:
            pickle.dump(self.parameters, file)
            print(f'Parameters saved at {self.save_loc}/parameters.pickle')
            
    def save_av(self):
        
        # Save as (nit, av) pairs
        nit = self.mj.nit
        av = self.mj.res.fun
        self.av_list.append((nit, av))
        
        # Pickle av_list
        with open(self.save_loc + '/av.pickle','wb') as file:
            pickle.dump(self.av_list, file)
            print(f'Action values saved at {self.save_loc}/av.pickle')
            
    def save_status(self):
        
        # Info we want
        success = self.mj.res.success
        message = self.mj.res.message
        nit = self.mj.nit
        
        # Write to text file
        f = open(self.save_loc + "/status.txt", "w")
        f.write(f'{nit} iterations completed\nSuccess: {success}\n{message}\n')
        print(f'Converged: {success}')
        f.close()
    
    def snapshot(self):
        "Saves instanton, status and action value."
        
        # Put Instanton in xr form
        dic = {}
        _time = self.mj.time
        _X = self.mj.instanton[:, :-1]
        _T = self.mj.instanton[:, -1]
        dic['X'] = xr.DataArray(_X, dims=['time', 'space'], name='X',
                            coords = {'time': _time,'space': np.arange(1, 1 + self.parameters['K'])})
        dic['T'] = xr.DataArray(_T, dims=['time'], name='T',
                                coords = {'time': _time})
        
        instanton = xr.Dataset(dic, attrs= self.parameters)
        
        # Save Instanton
        nit = self.mj.nit
        instanton.to_netcdf(self.save_loc + f'/instanton.nc')
        print('Instanton saved at ' + self.save_loc + f'/instanton.nc')

        # Save Action Value and current minimisation status
        self.save_av()
        self.save_status()
        return