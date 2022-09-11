
# Everyone needs numpy in their life:
import numpy as np
from numpy import pi
# For typing hints
import typing as typ
# For plotting:
import matplotlib.pyplot as plt
# For tools and helpers:
from utils import decorators
from archive.variables_and_units import Variable, Units
from dataclasses import dataclass


@dataclass
class Constants():
    h_bar   : float = 1.05457182*np.power(10,-34)  # m^2 Kg / s

@dataclass
class Params():
    # Given Params:
    N       : int   = 8
    Gamma   : float = 1
    Omega_0 : float = 0.01
    omega_0 : float = 10
    tMax    : float = 2
    dt      : float = 0.01

    # Derived Params:
    @property
    def J(self) -> float:        
        J = int(self.N/2)
        return J

    @property
    def mu(self) -> float:        
        mu = 3*self.Omega_0 / (8 * np.pi )
        return mu

    @property
    def Tr(self) -> float:        
        """Tr Characteristics superradiance time
        """
        res = 1/(self.N*self.Gamma*self.mu)
        return res

    @property
    def avr_tD(self) -> float:         
        """avr_tD average pulse delay
        """
        res = 1/(self.N*self.Gamma*self.mu)
        return res

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

def _intensity( theta:float, params:Params=Params() ) -> float :
    # Parse Values:
    omega_0 = params.omega_0
    N = params.N
    Tr = params.Tr
    intensity = (omega_0/2)*(N/Tr)*np.square(np.sin(theta)) # times h_bar
    return intensity 

def _theta_evolution( theta_i:float, t:float, params:Params=Params() ) -> float:
    # Parse constants:
    N = params.N
    Tr = params.Tr
    mu = params.mu
    # Compute:
    tg_theta_over2 = np.tan(theta_i/2)*np.exp()

# @Decorators.timeit
def main( params:Params=Params() ):    
    alpha = 0.5
    theta_i, phi_i = _initial_conditions(alpha, params)
    Tr = params.Tr

if __name__ == "__main__":    
    main()
    print("Done")