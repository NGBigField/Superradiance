# ==================================================================================== #
#|                                    Imports                                         |#
# ==================================================================================== #
import numpy as np
from scipy.linalg import sqrtm
from utils import assertions

# ==================================================================================== #
#|                                   Constants                                        |#
# ==================================================================================== #
EPS = 1e-10

# ==================================================================================== #
#|                                  Helper Types                                      |#
# ==================================================================================== #
_DensityMatrixType = np.matrix

# ==================================================================================== #
#|                                 Inner Functions                                    |#
# ==================================================================================== #


# ==================================================================================== #
#|                                Declared Functions                                  |#
# ==================================================================================== #
def distance(rho:_DensityMatrixType, sigma:_DensityMatrixType) -> float:
    diff = np.linalg.norm(rho - sigma)
    return diff**2
        
def fidelity(rho:_DensityMatrixType, sigma:_DensityMatrixType) -> float:
    root_rho = sqrtm(rho)
    root_prod = sqrtm(root_rho@sigma@root_rho)
    tr = np.trace(root_prod)
    fidel = assertions.real(tr**2, reason="Fidelity should be a real number") 
    return fidel

def purity(rho:_DensityMatrixType) -> float:
    """purity purity measure of quantum state as density matrix.

        purity of state = trace(rho^2):
        1 - if pure
        1/N - maximally not pure 

    Args:
        rho (_DensityMatrixType): quantum state

    Returns:
        float: purity of rho
    """
    rho2 = rho@rho
    tr = np.trace(rho2)
    pur = assertions.real(tr, reason="Purity should be a real number")
    return pur

def negativity(rho:_DensityMatrixType) -> float:
    raise NotImplementedError()

# ==================================================================================== #
#|                                     Tests                                          |#
# ==================================================================================== #

def _main_tests():
    from fock import Fock
    rho = Fock(0).to_density_matrix(num_moments=2)

if __name__ == "__main__" :
    _main_tests()