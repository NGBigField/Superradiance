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
        assert _num_M_vals(self) == N + 1

def _num_M_vals(params: Params) -> int:
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

def iterate_rho(params: Params, rho: np.matrix) -> typ.Generator[ typ.Tuple[int, float, float], None, None]:
    # Check Inputs:
    num_M = _num_M_vals(params)
    assert num_M == rho.shape[0]
    assert num_M == rho.shape[1]
    # Iterate on following m indices:
    for m in range(num_M):
        rho_m = rho[m, m]
        if m + 1 >= num_M:
            rho_m_plus_1 = 0
        else:
            rho_m_plus_1 = rho[m+1, m+1]
        yield m, rho_m, rho_m_plus_1

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

def init_state(params:Params, initial_state:CommonStates=CommonStates.FullyExcited) -> np.matrix:
    numM = _num_M_vals(params)
    rho = np.zeros([numM, numM])
    if initial_state==CommonStates.FullyExcited:
        rho[-1,-1] = 1
    elif initial_state==CommonStates.Ground:
        rho[0,0] = 1
    return rho

def evolve(
    params  : Params,
    rho_prev : np.matrix,  # previous density matrix 
) -> np.matrix:
    # Parse params:
    Gamma   = params.Gamma
    J       = params.J
    dt      = params.dt
    # init output:
    rho_next = np.zeros(rho_prev.shape)
    # Iterate:
    for m, rho_prev_m, rho_prev_mp1 in iterate_rho(params, rho_prev):
        M = _m2M(params, m)
        d_rho = -Gamma*(J+M)*(J-M+1)*rho_prev_m + Gamma*(J-M)*(J+M+1)*rho_prev_mp1
        rho_next_m = dt*d_rho + rho_prev_m
        # Insert Values:
        rho_next[m] = rho_next_m
    return rho_next

def evolve(
    params  : Params,
    rho_prev : np.array,  # previous density matrix 
) -> np.array:
    # Parse params:
    Gamma   = params.Gamma
    J       = params.J
    dt      = params.dt
    # init output:
    rho_next = np.zeros(rho_prev.shape)
    # Iterate:
    for m, rho_prev_m, rho_prev_mp1 in iterate_rho(params, rho_prev):
        M = _m2M(params, m)
        d_rho = -Gamma*(J+M)*(J-M+1)*rho_prev_m + Gamma*(J-M)*(J+M+1)*rho_prev_mp1
        rho_next_m = dt*d_rho + rho_prev_m
        # Insert Values:
        rho_next[m,m] = rho_next_m
    return rho_next

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
    numM = _num_M_vals(params)
    psi_i = init_state(params, CommonStates.Ground)
    psi_f = op@psi_i

# ==================================================================================== #
#                                   main()                                             #
# ==================================================================================== #

def _main_test( params:Params=Params() ):    

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
    _main_test()
    # coherent_pulse()
    print("Done")