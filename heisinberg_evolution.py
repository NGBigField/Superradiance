
# Everyone needs numpy in their life:
import numpy as np
from numpy import pi
# For typing hints
import typing as typ
# For plotting:
import matplotlib.pyplot as plt
# For tools and helpers:
from utils import Decorators
from dataclasses import dataclass


@dataclass
class Params():
    # Given Params:
    N       : int   = 8
    Gamma   : float = 1
    Omega_0 : float = 0.01
    tMax    : float = 2
    dt      : float = 0.01

    # Derived Params:
    @property
    def J(self):        
        J = int(self.N/2)
        return J

    @property
    def mu(self):        
        mu = 3*self.Omega_0 / (8 * np.pi )
        return mu

    @property
    def Tr(self):        
        Tr = 1/(self.N*self.Gamma*self.mu)
        return Tr

    def validate(self) -> None:
        N = self.N
        J = self.J
        assert float(N)/2 == int(int(N)/2) # N is even
        assert J*2 == N 
        assert _numMVals(self) == N + 1

def _numMVals(params: Params) -> int:
    N = params.N
    return N+1

def _initial_conditions(alpha:float, params:Params=Params()):
    theta_i = alpha*2/(params.N)
    phi_i = np.random.uniform(low=0, high=2*pi) # Answer in radians
    return theta_i, phi_i

@Decorators.timeit
def main( params:Params=Params() ):    
    alpha = 0.5
    theta_i, phi_i = _initial_conditions(alpha, params)

if __name__ == "__main__":    
    main()
    print("Done")