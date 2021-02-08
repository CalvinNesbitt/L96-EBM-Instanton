"""
Contains bits of code that we will repeatedly use for our instanton work.

-------------------------------------------------------
Contents
-------------------------------------------------------

-----------------
Classes
-----------------

action_minimisation_checker

    Class for keeping track of instanton action comparisons.
    
-----------------
Functions
-----------------
eps_path
    Takes path and perturbs all non boundary points by constant.

random_path
    Takes path and adds a random number to each point.

"""

import numpy as np
import numpy.random as rm


#--------------------------------------------------------------------
# Classes
#--------------------------------------------------------------------

class action_minimisation_checker:
    """
    Class for keeping track of instanton action comparisons.
    
    - User provides 'instanton' and SDE drift on object creation. 
    - .compare(path) method can then be used to compute F-W action value for nearby paths.
    - Object keeps track of compared paths (see properties)
    
    Properties
    -----------
    compared_paths
        List of tuples. Tuples of form (path, FW action value). 
        Includes initial instanton.
    
    path_list
        List of compared paths. 
        Includes initial instanton.
        
    av_list
        List of FW action values for compared paths. 
        Includes initial instanton.
        
    smaller_list
        List of tuples for paths with smaller FW action value. 
        Tuples of form (path, FW action value).
        
    Methods
    -----------
    compare(path)
        Calculates FW action for provided path and adds to comparison list.
        
    any_smaller()
        Checks if any paths in the comparison list have a smaller action.
        
    Attributes
    -----------
    instanton
        The path provided on object creation. Meant to be instanton we're comparing against.
        
    action_value
        FW action value for instanton provided on object creation.
        
    b_args
        Arguments used in SDE drift for FW action computation.
    
    """
    
    def __init__(self, a, instanton, args):
        """ 
         Parameters
        ----------
        a: function 
            Takes path and returns drift. 
            Should be of form a(path, a_args).
            Path input and are drift output are shape (time, ndim).
        
        instanton: numpy array
            The path/instanton we're comparing against.
            
        args: list
            List of form [a_args, time, d_inv]
            d_inv is inverse of diffusiong matrix
        """
        self._b = a
        self.b_args, self.time, self.d = args
        
        # Instanton we will we check against
        self.instanton = instanton 
        
        if (len(self.instanton.shape) == 1): #ndim=1 case
            self.action_value = self._1d_action(instanton)
        else:
            self.action_value = self._action(instanton)

        
        self._compare_list = [(self.instanton, self.action_value)]
        self._smaller_list = []
        
    def _action(self, path):
        """ Takes a path and computes the FW action along the path when ndim >1. If ndim=1,
        object will use _1d_action.

        Path input is of form necesseary for scipy.minimise (see parameter explanation).
        Finite differences are used to approximate derivatives.
        Trapezoidal Rule used to compute integral

        Parameters
        ----------
        path: np array
            shape is flat (time * ndim) for use by scipy.minimise
            method reshapes into (time, ndim)
        """
        v = np.vstack(np.gradient(path, self.time.flatten(), 1)[0]) - self._b(0, path, self.b_args)

        # Dot product calulcation
        v2 = [] 
        for x in v:
            v2.append(x.dot(self.d @ x.T))
        return 0.5 * np.trapz(v2, x=self.time)
    
    def _1d_action(self, path):
        """
        Same as _action method for 1d case.
        """
        v = np.gradient(path, self.time) -  self._b(0, path, self.b_args)
        integrand =  v**2
        return 0.5 * np.trapz(integrand, x=self.time)
    
    def compare(self, path):
        """
        Calculates FW action for provided path and adds to comparison list.
        """
        if (len(self.instanton.shape) == 1): #ndim=1 case
            x = self._1d_action(path)
        else:
            x = self._action(path)
        self._compare_list.append((path, x))
    
    @property
    def compared_paths(self):
        "Returns a list of tuples of compared paths and their action value"
        return self._compare_list
    
    @property
    def path_list(self):
        "List of action values from compared paths"
        return [x[1] for x in self.compared_paths]
    
    @property
    def av_list(self):
        "List of action values from compared paths"
        return [x[1] for x in self.compared_paths]
        
    def any_smaller(self):
        "Checks if any compared path has a smaller action."
        for path_tuple in self.compared_paths:
            if (path_tuple[1] < self.action_value):
                print("Looks like a nearby path has a smaller action")
                self._smaller_list.append(x)
                print("I've made a note of this")
                return 
        print("Looks like it minimises")
        return
    
    @property
    def smaller_list(self):
        if len(self._smaller_list == 0):
            self.any_smaller()
        return self._smaller_list
    
#--------------------------------------------------------------------
# Functions
#--------------------------------------------------------------------
    
def eps_path(path, x):
    "Function that perturbs all non boundary points for a given path by x."
    ep = np.copy(path)
    ep[1:-1] += x
    return ep

def random_path(path, b):
    """Takes path and adds a random number to each point.\
    For each point on the path an number is sampled from uniform(-b, b)
    and added to that point.
    """
    return path + rm.uniform(-b, b, path.shape)