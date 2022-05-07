
# Everyone needs numpy in their life:
import numpy as np
# For typing hints
import typing as typ
# For plotting:
import matplotlib.pyplot as plt
# For tools and helpers:
from Utils import Decorators

""" 
# TODO:
1. Rho can be a vector. we only use diagonal values
2. m <-  0 to N instead of using M
3. maybe solve in steps as we do, or solve as matrix exponent

"""

     

# Constants:
N = 6
Gamma = 1
tau  = 0 # time of emission =  (1/Gamma*N)
tMax = 0.4
dt = 0.1


# Derived Params:
J = int(N/2)

def _CheckConstants() -> None:
    assert float(N)/2 == int(int(N)/2) # N is even
    assert J*2 == N 
    assert _numMVals() == N + 1

def TimeIterator() -> typ.Iterator:
    t = 0.00
    while t <= tMax:
        yield t
        t += dt

def RhoIterator(rho: np.array) -> typ.Iterator[typ.Tuple[int, float, float]]:
    numM = _numMVals()
    assert numM == len(rho)
    for m in range(numM):
        rho_m = rho[m]
        if m + 1 >= numM:
            rho_mPlus1 = 0
        else:
            rho_mPlus1 = rho[m+1]
        yield (m, rho_m, rho_mPlus1)

@Decorators.assertType(int)
def _M2m(M: int) -> int:
    m = M+int(N/2)

@Decorators.assertType(int)
def _m2M(m: int) -> int:
    M = m-int(N/2)

def Energy(rho: np.matrix) -> float:
    """ 
       E_Vec = rho_vec  \cdot   (0:N)
    """
    Sum = 0.0
    for i, M in enumerate(_allMVals()):
        m = _M2m(M)
        element = rho[i,i]*m
        Sum += element
    return Sum

def Intensity( energyVec, timeVec ) -> np.ndarray:
    """ 
    - dE/dt  = -sum_M(  drho/dt  * m ) = - (H*rho_vec .*  E_Vec)
    drho_Vec/dt = H*rho_vec   a matrix representation of the equation (4.7)
    """
    L = len(energyVec)
    assert len(timeVec)==L
    intensityVec = np.ndarray((L-1), dtype=float)
    for i in range(L-1):
        intensityVec[i] = (-1)*(energyVec[i+1]-energyVec[i])/(timeVec[i+1]-timeVec[i])
    return intensityVec


def _allMVals() -> typ.Iterable:
    Iter = range(-J,J+1,1)
    return Iter

def _numMVals() -> int:
    return N+1

def InitRho() -> np.array:
    numM = _numMVals()
    rho = np.zeros([numM])
    rho[-1] = 1
    return rho

def Evolve(
    prevRho : np.array,  # previous density matrix 
    dt      : float,     # size of time steps
) -> np.array:

    # init output:
    nextRho = np.zeros(prevRho.shape)
    # Iterate:
    for m, prevRho_m, prevRho_mPlus1 in RhoIterator(prevRho):
        M = _m2M(m)
        dRho = -Gamma*(J+M)*(J-M+1)*prevRho_m + Gamma*(J-M)*(J+M+1)*prevRho_mPlus1
        nextRho_m = dt*dRho + prevRho_m
        # Insert Values:
        nextRho[m] = nextRho_m
    return nextRho
    
@Decorators.timeit
def main():    

    # Check:
    _CheckConstants()

    # Init:
    rho = InitRho()
    energyVec = []
    timeVec   = []
    
    # Simulate Evolution:
    for t in TimeIterator():
        rho = Evolve(rho, dt)
        E = Energy(rho)
        timeVec.append( t )
        energyVec.append( E )

    # Compute Intensity:
    intensityVec = Intensity(energyVec, timeVec)

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


if __name__ == "__main__":
    main()
    print("Done")