# Everyone needs numpy and scipy in their life:
import numpy as np
from scipy.linalg import expm as matrix_exp

# For typing hints
import typing as typ
from dataclasses import dataclass
from enum import Enum, auto

# our modules, tools and helpers:
from visuals import (
    plot_superradiance_evolution
)


# ==================================================================================== #
#                                   config                                             #
# ==================================================================================== #

@dataclass
class Params():
    # Given Params:
    N       : int   = 8
    Gamma   : float = 1
    tMax    : float = 2
    dt      : float = 0.01

    # Derived Params:
    @property
    def J(self):        
        J = int(self.N/2)
        return J

    def validate(self) -> None:
        N = self.N
        J = self.J
        assert float(N)/2 == int(int(N)/2) # N is even
        assert J*2 == N 
        assert _numMVals(self) == N + 1

def _numMVals(params: Params) -> int:
    N = params.N
    return N+1

# ==================================================================================== #
#                                 helper functions                                     #
# ==================================================================================== #

def iterate_time(params: Params) -> typ.Iterator:
    tMax = params.tMax
    dt   = params.dt
    t = 0.00
    while t <= tMax:
        yield t
        t += dt

def iterate_rho(params: Params, rho: np.array) -> typ.Iterator[typ.Tuple[int, float, float]]:
    numM = _numMVals(params)
    assert numM == len(rho)
    for m in range(numM):
        rho_m = rho[m]
        if m + 1 >= numM:
            rho_mPlus1 = 0
        else:
            rho_mPlus1 = rho[m+1]
        yield (m, rho_m, rho_mPlus1)

def _m2M(params: Params, m: int) -> int:
    N = params.N
    M = m-int(N/2)
    return M

def calc_energy(params: Params, rho: np.array) -> float:
    """ 
       E_Vec = rho_vec  \cdot   (0:N)
    """
    Sum = 0.0
    for m, rho_m, _ in iterate_rho(params, rho):
        Sum += rho_m*m
    return Sum

def calc_intensity( energyVec: list, timeVec: list ) -> np.ndarray:
    """ 
    - dE/dt  = -sum_M(  drho/dt  * m ) = - (H*rho_vec .*  E_Vec)
    drho_Vec/dt = H*rho_vec   a matrix representation of the equation (4.7)
    """
    # Check Inputs:
    L = len(energyVec)
    assert len(timeVec)==L
    # Init Outputs:
    intensityVec = np.ndarray((L-1), dtype=float)
    # Calculate
    for i in range(L-1):
        intensityVec[i] = (-1)*(energyVec[i+1]-energyVec[i])/(timeVec[i+1]-timeVec[i])
    return intensityVec

class CommonStates(Enum):
    Ground = auto()
    FullyExcited = auto()

def init_state(params:Params, initial_state:CommonStates=CommonStates.FullyExcited) -> np.array:
    numM = _numMVals(params)
    rho = np.zeros([numM])
    if initial_state==CommonStates.FullyExcited:
        rho[-1] = 1
    elif initial_state==CommonStates.Ground:
        rho[0] = 1
    return rho

def evolve(
    params  : Params,
    prevRho : np.array,  # previous density matrix 
) -> np.array:
    # Parse params:
    Gamma   = params.Gamma
    J       = params.J
    dt      = params.dt
    # init output:
    nextRho = np.zeros(prevRho.shape)
    # Iterate:
    for m, prevRho_m, prevRho_mPlus1 in iterate_rho(params, prevRho):
        M = _m2M(params, m)
        dRho = -Gamma*(J+M)*(J-M+1)*prevRho_m + Gamma*(J-M)*(J+M+1)*prevRho_mPlus1
        nextRho_m = dt*dRho + prevRho_m
        # Insert Values:
        nextRho[m] = nextRho_m
    return nextRho

def coherent_pulse(params:Params=Params()):
    # Inputs:

    # Constants:
    hbar = 1
    Sx = np.matrix([[0,1],[1,0]])

    # Omega:
    Omega = 1

    # Create the mat:
    mat = (1j/hbar)*Omega*Sx
    op = matrix_exp(mat)

    # Operator on state
    numM = _numMVals(params)
    psi_i = init_state(params, CommonStates.Ground)
    psi_f = op@psi_i

# ==================================================================================== #
#                                   main()                                             #
# ==================================================================================== #

def main( params:Params=Params() ):    

    # Check:
    params.validate()

    # Init:
    rho = init_state(params)
    energies = []
    times   = []
    
    # Simulate Evolution:
    for t in iterate_time(params):
        # Evolve:
        rho = evolve(params, rho)
        E = calc_energy(params, rho)
        # Keep results in Vectors:
        times.append( t )
        energies.append( E )

    # Compute Intensity:
    intensities = calc_intensity(energies, times)

    # Plot Results:
    plot_superradiance_evolution(times, energies, intensities)


if __name__ == "__main__":
    # main()
    coherent_pulse()
    print("Done")