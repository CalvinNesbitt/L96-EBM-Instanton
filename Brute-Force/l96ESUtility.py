"""
Contains code needed to run the stochastic L96 EBM using diffeqpy.
"""

# diffeqpy requires Julia
from julia.api import Julia
jl = Julia(compiled_modules=False)
from diffeqpy import de
import numba

# Standard Package imports
import numpy as np
import numpy.linalg as la
import numpy.random as rm
import xarray as xr
import matplotlib.pyplot as plt


# ------------------------------------------
# Convenience Class for Observing Solution
# ------------------------------------------

class SolutionObserver():
    """
    Observes the solution of L96 EBM stochastic integrator. Dumps to netcdf.
    NOTE IT IS DIFFERENT FROM OUR SCIPY based OBJECTS.
    """

    def __init__(self, p, name='L96-EBM-Stochastic-Trajectory'):
        """param, p, list: parameters of this run."""
        
        self.name = name
        self.dump_count = 0

        # Needed knowledge of the integrator
        keys = ['K','S','a0','a1','sigma','F','Tref','delT','alpha','beta', 'eps', 'delta']
        self._parameters = dict(zip(keys, p))
        self._K = p[0]
        
        # Trajectory Observation logs
        self.time_obs = [] # Times we've made observations
        self.u_obs = []

    def look(self, sol):
        """Observes trajectory by looking at solution object"""

        # Note the time
        self.time_obs += list(sol.t)

        # Making Observations
        self.u_obs += list(sol.u)
        return

    @property
    def observations(self):
        """cupboard: Directory where to write netcdf."""
        if (len(self.u_obs) == 0):
            print('I have no observations! :(')
            return

        dic = {}
        _time = self.time_obs
        _T = np.stack(self.u_obs)[:, -1]
        _X = np.stack(self.u_obs)[:, :-1]
        dic['X'] = xr.DataArray(_X, dims=['time', 'space'], name='X',
                                coords = {'time': _time,'space': np.arange(1, 1 + self._K)})
        dic['T'] = xr.DataArray(_T, dims=['time'], name='T',
                                coords = {'time': _time})
        return xr.Dataset(dic, attrs= self._parameters)

    def wipe(self):
        """Erases observations"""
        self.time_obs = []
        self.u_obs = []

    def dump(self, cupboard, name=None):
        """ Saves observations to netcdf and wipes.
        cupboard: Directory where to write netcdf.
        name: file name"""

        if (len(self.u_obs) == 0):
            print('I have no observations! :(')
            return

        if name == None:
            name=self.name

        save = cupboard + f'/{name}' + f'{self.dump_count + 1}.nc'
        self.observations.to_netcdf(save)
        print(f'Observations written to {save}. Erasing personal log.\n')
        self.wipe()
        self.dump_count +=1
        

