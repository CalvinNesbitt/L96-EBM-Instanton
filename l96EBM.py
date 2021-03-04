""" Lorenz 96 EBM  Integrator classes.
Based on equations 3 of Gelbrecht et al. 2020 paper (https://arxiv.org/abs/2011.12227).
Uses adaptive RK54 method.
----------
Contents
----------
- Integrator, class for integrating L96 two layer dynamics.

- TrajectoryObserver, class for observing the trajectory of the L96 integration.

- make_observations, function that makes many observations given integrator and observer.
"""

# ----------------------------------------
# Imports
# ----------------------------------------
import numpy as np
import scipy.integrate
import xarray as xr
import sys
from tqdm import tqdm

# ------------------------------------------
# Integrator
# ------------------------------------------

class Integrator:

    """Integrates the 1 layer L96 model, coupled to EBM."""
    def __init__(self, p=[40, 12, 0.5, 0.4, 1/180**4, 8, 270, 60, 2, 1], X_init=None, T_init=None):

        # Model parameters
        [self.K, self.S, self.a0, self.a1, self.sigma,
        self.F, self.Tref, self.delT, self.alpha, self.beta] = p
            
        self.size = self.K + 1 # Number of variable
        self.time = 0

        # Non-linear Variables
        self.X = np.random.rand(self.K) if X_init is None else X_init.copy() # Random IC if none given
        self.T = np.random.randn(1) + 270 if T_init is None else T_init.copy()  


    def _rhs_X_dt(self, X, T):
        """Compute the right hand side of the X-ODE."""

        dXdt = (
                np.roll(X, 1) * (np.roll(X, -1) - np.roll(X, 2)) -
                X + self.F * (1 + self.beta * (T - self.Tref)/self.delT )
        )
        return dXdt

    def _rhs_T_dt(self, X, T):
        """Compute the right hand side of the Y-ODE."""
        avg_energy = 0.5 * np.sum(X**2)/self.K
        dTdt = (
            self.S * (1 - self.a0 + 0.5 * self.a1 * (np.tanh(T - self.Tref))) 
            - self.sigma * T**4 - self.alpha * (avg_energy/(0.6 * self.F**(4/3)) - 1)
        )
        return dTdt


    def _rhs_dt(self, t, state):
        X, T = state[:self.K], state[self.K:]
        return [*self._rhs_X_dt(X, T), *self._rhs_T_dt(X, T)]

    def integrate(self, how_long):
        """time: how long we integrate for in adimensional time."""

        # Where We are
        t = self.time
        IC = self.state

        # Integration, uses RK45 with adaptive stepping. THIS IS THE HEART.
        solver_return = scipy.integrate.solve_ivp(self._rhs_dt, (t, t + how_long), IC, dense_output = True)

        # Updating variables
        new_state = solver_return.y[:,-1]
        self.X = new_state[:self.K]
        self.T = new_state[self.K: self.size]

        self.time = t + how_long

    def set_state(self, x):
        """x is [X, T]."""
        self.X = x[:self.K]
        self.T = x[self.K:]

    @property
    def state(self):
        """Where we are in phase space."""
        return np.concatenate([self.X, self.T])

    @property
    def time(self):
        """a-dimensional time"""
        return self.__time

    @time.setter
    def time(self, when):
        self.__time = when

    @property
    def parameter_dict(self):
        param = {
        'K': self.K,
        'S': self.S,
        'a0': self.a0, 
        'a1': self.a1,
        'sigma': self.sigma,
        'F': self.F,
        'Tref': self.Tref,
        'delT': self.delT,
        'alpha': self.alpha,
        'beta': self.beta
        }
        return param
    
# ------------------------------------------
# TrajectoryObserver
# ------------------------------------------

class TrajectoryObserver():
    """Observes the trajectory of L96 EBM integrator. Dumps to netcdf."""

    def __init__(self, integrator, name='L96-EBM-Trajectory'):
        """param, integrator: integrator being observed."""
        
        self.name = name
        self.dump_count = 0

        # Needed knowledge of the integrator
        self._parameters = integrator.parameter_dict
        self._K = integrator.K

        # Trajectory Observation logs
        self.time_obs = [] # Times we've made observations
        self.x_obs = []
        self.t_obs = []

    def look(self, integrator):
        """Observes trajectory of L96 trajectory"""

        # Note the time
        self.time_obs.append(integrator.time)

        # Making Observations
        self.x_obs.append(integrator.X.copy())
        self.t_obs.append(*integrator.T.copy())
        return

    @property
    def observations(self):
        """cupboard: Directory where to write netcdf."""
        if (len(self.x_obs) == 0):
            print('I have no observations! :(')
            return

        dic = {}
        _time = self.time_obs
        dic['X'] = xr.DataArray(self.x_obs, dims=['time', 'space'], name='X',
                                coords = {'time': _time,'space': np.arange(1, 1 + self._K)})
        dic['T'] = xr.DataArray(self.t_obs, dims=['time'], name='T',
                                coords = {'time': _time})
        return xr.Dataset(dic, attrs= self._parameters)

    def wipe(self):
        """Erases observations"""
        self.time_obs = []
        self.x_obs = []
        self.t_obs = []

    def dump(self, cupboard, name=None):
        """ Saves observations to netcdf and wipes.
        cupboard: Directory where to write netcdf.
        name: file name"""

        if (len(self.x_obs) == 0):
            print('I have no observations! :(')
            return

        if name == None:
            name=self.name

        save = cupboard + f'/{name}' + f'{self.dump_count + 1}.nc'
        self.observations.to_netcdf(save)
        print(f'Observations written to {save}. Erasing personal log.\n')
        self.wipe()
        self.dump_count +=1
        
# ------------------------------------------
# make_observations
# ------------------------------------------

def make_observations(runner, looker, obs_num, obs_freq, noprog=True):
    """Makes observations given runner and looker.
    runner, integrator object.
    looker, observer object.
    obs_num, how many observations you want.
    obs_freq, adimensional time between observations"""
    for step in tqdm(np.repeat(obs_freq, obs_num), disable=noprog):
        runner.integrate(obs_freq)
        looker.look(runner)