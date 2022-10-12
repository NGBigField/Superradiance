# ==================================================================================== #
#|                                    Imports                                         |#
# ==================================================================================== #
import numpy as np
from scipy.linalg import sqrtm

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
def simmilarity(rho:_DensityMatrixType, sigma:_DensityMatrixType) -> float:
    diff = np.linalg.norm(rho - sigma)
    return diff**2
        
def fidelity(rho:_DensityMatrixType, sigma:_DensityMatrixType) -> float:
    root_rho = sqrtm(rho)
    root_prod = sqrtm(root_rho@sigma@root_rho)
    tr = np.trace(root_prod)
    return tr**2

def purity(rho:_DensityMatrixType) -> float:
    tr = np.trace(rho@rho)
    assert np.imag(tr)<EPS, f"Purity should be a real number"
    return np.real(tr)

# ==================================================================================== #
#|                                     Tests                                          |#
# ==================================================================================== #

def _main_tests():
    from fock import Fock
    rho = Fock(0).to_density_matrix(num_moments=2)

if __name__ == "__main__" :
    _main_tests()