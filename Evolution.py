
# Everyone needs numpy in their life:
import numpy as np
# For typing hints
import typing as typ
# calc observables
import Observables


# Constants:
J = 3 # Use 20
Gamma = 1
dt = 1



def Energy(rho: np.matrix) -> float:
    Sum = 0.0
    for i, M in enumerate(_allMVals()):
        element = rho[i,i]*M
        Sum += element
    return Sum

def Intensity( energyVec, timeVec ) -> np.ndarray:
    L = len(energyVec)
    assert len(timeVec)==L
    intensityVec = np.zeros((L-1),1)
    for i in range(L-1):
        intensityVec[i] = (-1)*(energyVec(i+1)-energyVec(i))/(timeVec(i+1)-timeVec(i))
    return intensityVec




def _allMVals() -> iter:
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
    


if __name__ == "__main__":
    rho = InitDensityMatrix()

    timeVec = list(range(0,10,dt))
    rhoList = list()
    energyVec = np.ndarray((len(timeVec), 1), dtype=float)
    for i, t in enumerate(timeVec):
        rho = EvolveDensityMatrix(rho, dt)
        rhoList.append(rho)
        energyVec[i] = Observables.Energy(rho)
    print("Done.")