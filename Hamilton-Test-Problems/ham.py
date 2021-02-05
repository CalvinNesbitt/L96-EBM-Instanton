# Object that performs Grafke Algorithm

# Standard Package imports

import numpy as np
import numpy.random as rm
import scipy.integrate

from scipy.optimize import approx_fprime
from tqdm.notebook import tqdm

# Object that performs Grafke Algorithm

# Object that performs Grafke Algorithm

class Hamilton_solver:
    
    """Performs Grafke 2019 Section IIIA Hamiltonian Algorithm to compute instantons.
    Main use is through .run() method"""
    def __init__(self, rhs, time, IC, s, update):
        """
        - rhs=[phi_rhs, theta_rhs] defines the Hamilton ODE. 
        - time are the points the Instanton is parameterised over
        - IC is shape (time, 2 * ndim)
        - s are parameters for the rhs. For phi integrator should be [p, theta]. For theta integrator [p, phi]
        - update=[F, lamb] are theta update choices 
        """
        
        # Object self awareness
        self.ndim = int(IC.shape[1]/2)
        self.where = len(time) - 1 # Keeps track of where we are in time
        self.step_count = 1 # How many loops of algorithm
        
        # Unpacking Input
        self.F, self.lamb = update
        self. s = s
        self.phi_rhs, self.theta_rhs = rhs
        self.time = time
        
        # Instanton 
        self.__phi_ts, self.__theta_ts = IC[:, :self.ndim], IC[:, self.ndim:] 
        return
    
    @property
    def phi_ts(self):
        return self.__phi_ts
    
    @phi_ts.setter   
    def phi_ts(self, x):
        self.__phi_ts = x
        
    @property
    def theta_ts(self):
        """a-dimensional time"""
        return self.__theta_ts
    
    @theta_ts.setter   
    def theta_ts(self, x):
        self.__theta_ts = x
    
    @property
    def instanton(self):
        "Returns current Instanton"
        return np.hstack((self.phi_ts, self.theta_ts))
        
    # Integrators
    
    def _phi_step(self):
        "One step of forward integration"
        
        # Unpack
        if (self.where == len(self.time) - 2):
            start, end = self.time[len(self.time) - 2 :]
        else:
            start, end = self.time[self.where : self.where + 2]
            
        state = self.phi_ts[self.where] # This should only be phi variables, theta variables treated as argument!
        current_theta = self.theta_ts[self.where]
        
        # Integrate Phi Equation
        result = scipy.integrate.solve_ivp(self.phi_rhs, (start, end), state, 
                                           args=[[self.s, current_theta]], dense_output = True)
        
       # Update
        self.where += 1
        self.phi_ts[self.where] = result.y[:, -1]
        return 
                
    def _theta_step(self):
        "One step of backward integration"
        
        # Unpack
        if (self.where == 1):
            start, end = np.flip(self.time[:2])
        else:
            start, end = self.time[self.where: self.where - 2: -1]
        
        state = self.theta_ts[self.where] # This should only be theta variables!
        current_phi = self.phi_ts[self.where]
        
        # Integrate Theta Equation
        result = scipy.integrate.solve_ivp(self.theta_rhs, (start, end), state, 
                                           args=[[self.s, current_phi]], dense_output = True)
        
        # Update
        self.where += -1
        self.theta_ts[self.where] = result.y[:, -1]
        return
    
    def _backwards_integration(self):
        for t in self.time[:-1]:
            self._theta_step()
    
    def _forwards_integration(self):
        for t in self.time[:-1]:
            self._phi_step()

    def _theta_IC_update(self):
        "Remember theta equation is solved backwards so last point is IC"\
        #Finite differences used to approximate spatial derivative, may be issues due to it being an end point
        numerical_gradient = approx_fprime(self.__phi_ts[-1], self.F, 0.01)# scipy function uses finite differences
        self.__theta_ts[-1,:] = -self.lamb * numerical_gradient
                 
    def _alg_step(self):
        "One step of the algorithm"
        self._backwards_integration()
        self._forwards_integration()
        self._theta_IC_update()
        
        self.step_count +=1
        
    def run(self, steps, timer=True):
        "Run Algorithm for specified number of steps"
        for i in tqdm(range(int(steps)), disable = not timer):
            self._alg_step()
            
            