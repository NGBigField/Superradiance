
# Everyone needs numpy in their life:
import numpy as np
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

def TimeIterator(params: Params) -> typ.Iterator:
    tMax = params.tMax
    dt   = params.dt
    t = 0.00
    while t <= tMax:
        yield t
        t += dt

def RhoIterator(params: Params, rho: np.array) -> typ.Iterator[typ.Tuple[int, float, float]]:
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

def Energy(params: Params, rho: np.array) -> float:
    """ 
       E_Vec = rho_vec  \cdot   (0:N)
    """
    Sum = 0.0
    for m, rho_m, _ in RhoIterator(params, rho):
        Sum += rho_m*m
    return Sum

def Intensity( energyVec: list, timeVec: list ) -> np.ndarray:
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


def InitRho(params: Params) -> np.array:
    numM = _numMVals(params)
    rho = np.zeros([numM])
    rho[-1] = 1
    return rho

def Evolve(
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
    for m, prevRho_m, prevRho_mPlus1 in RhoIterator(params, prevRho):
        M = _m2M(params, m)
        dRho = -Gamma*(J+M)*(J-M+1)*prevRho_m + Gamma*(J-M)*(J+M+1)*prevRho_mPlus1
        nextRho_m = dt*dRho + prevRho_m
        # Insert Values:
        nextRho[m] = nextRho_m
    return nextRho

def PlotResults(timeVec, energyVec, intensityVec):
    # Plot:
    fig, axes = plt.subplots(nrows=2, ncols=1, constrained_layout=True)

    axes[0].plot(timeVec[0:-1], energyVec[0:-1])    
    axes[0].grid(which='major')
    axes[0].set_xlabel('Time [sec] '    , fontdict=dict(size=16) )
    axes[0].set_ylabel('Energy     '    , fontdict=dict(size=16) )
    axes[0].set_title('Evolution   '    , fontdict=dict(size=16) )

    axes[1].plot(timeVec[0:-1], intensityVec)    
    axes[1].grid(which='major')
    axes[1].set_xlabel('Time [sec] '    , fontdict=dict(size=16) )
    axes[1].set_ylabel('Intensity  '    , fontdict=dict(size=16) )
    plt.show()

@Decorators.timeit
def main( params: Params ):    

    # Check:
    params.validate()

    # Init:
    rho = InitRho(params)
    energyVec = []
    timeVec   = []
    
    # Simulate Evolution:
    for t in TimeIterator(params):
        # Evolve:
        rho = Evolve(params, rho)
        E = Energy(params, rho)
        # Keep results in Vectors:
        timeVec.append( t )
        energyVec.append( E )

    # Compute Intensity:
    intensityVec = Intensity(energyVec, timeVec)

    # Plot Results:
    PlotResults(timeVec, energyVec, intensityVec)


if __name__ == "__main__":
    params = Params()
    main(params)
    print("Done")