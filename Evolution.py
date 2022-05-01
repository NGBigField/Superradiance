
# Everyone needs numpy in their life:
import numpy as np
# For typing hints
import typing as typ
# For plotting:
import matplotlib.pyplot as plt
# For tools and helpers:
from Utils import timeit


# Constants:
J = 20
Gamma = 1
dt = 0.5
nTimes = 20



def Energy(rho: np.matrix) -> float:
    Sum = 0.0
    for i, M in enumerate(_allMVals()):
        element = rho[i,i]*M
        Sum += element
    return Sum

def Intensity( energyVec, timeVec ) -> np.ndarray:
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
    return 2*J+1

def InitDensityMatrix() -> np.matrix:
    numM = _numMVals()
    rho = np.zeros([numM, numM])
    rho[-1,-1] = 1
    return rho

def EvolveDensityMatrix(
    prevRho : np.matrix, # previous density matrix 
    dt      : float,     # size of time steps
) -> np.matrix:

    # init output:
    nextRho = np.zeros(prevRho.shape)
    # Iterate:
    numM = _numMVals()
    for i, M in enumerate(_allMVals()):
        # Get values:
        prevRhoM = prevRho[i,i]
        if i==numM-1:
            prevRhoMPlus1 = 0
        else:
            prevRhoMPlus1 = prevRho[i+1,i+1]
        # Calc next rho:
        nextRhoM = -Gamma*(J+M)*(J-M+1)*prevRhoM + Gamma*(J-M)*(J+M+1)*prevRhoMPlus1
        nextRhoM *= dt
        # Insert Values:
        nextRho[i,i] = nextRhoM
    return nextRho
    

def main():    
    # Init:
    rho = InitDensityMatrix()
    timeVec = np.arange(start=0, stop=nTimes*dt,step=dt)
    rhoList = list()
    energyVec = np.zeros((len(timeVec)),dtype=float)   

    # Compute Evolution:
    for i, t in enumerate(timeVec):
        rho = EvolveDensityMatrix(rho, dt)
        rhoList.append(rho)
        energyVec[i] = Energy(rho)

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