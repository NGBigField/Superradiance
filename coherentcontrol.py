# ==================================================================================== #
# |                                 Imports                                          | #
# ==================================================================================== #

# Everyone needs numpy:
import numpy as np

# For typing hints:
from typing import (
    Tuple,
    List,
)

# import our helper modules
from utils import (
    assertions,
    numpy as np_utils
)

# For plotting test results:
import matplotlib.pyplot as plt


# ==================================================================================== #
# |                            Inner Functions                                       | #
# ==================================================================================== #

def _assert_N(N:int) -> None:
    assertions.even(N) # N must be even    

def _J(N:int) -> int : 
    _assert_N(N)
    return N//2

def _M(m:int, J:int) :
    return m-J

def _mat_size(N: int) -> Tuple[int, int]:
    return (N+1, N+1)

def _d_plus_minus(M:int, J:int, pm:int) -> float:
    """ _summary_

    N: even integer
    M: index from 0 to N (including. i.e. N+1 possible values)
    pm : +1 or -1

    return d plust\minus according to the relation:
    D^{\pm}\ket{J,M}=d^{\pm}\ket{J,M}==\sqrt{J(J+1)-M(M\pm1)}\ket{J,M\pm1}
    """
    d = np.sqrt(
        J*(J+1)-M*(M+pm)
    )
    return d

def _D_plus_minus_mats(N:int) -> Tuple[np.matrix, np.matrix] :
    # Derive sizes:
    mat_size = _mat_size(N)
    J = _J(N)
    # Init outputs:    
    D = {}
    D[+1] = np.zeros(mat_size)
    D[-1] = np.zeros(mat_size)
    # Iterate over diagonals:
    for m in range(mat_size[0]):
        M = _M(m, J)
        i = m

        for pm in [-1, +1]:
            # derive matrix indices:
            j = m + pm
            if j>=N+1 or j<0:
                continue
            # compute and assign to matrix
            d = _d_plus_minus(M, J, pm)
            D[pm][i,j] = d

    return D[+1], D[-1]

def _Sz_mat(N:int) -> np.matrix :
    mat_size = _mat_size(N)
    J = _J(N)
    Sz = np.zeros(mat_size)
    for m in range(mat_size[0]):
        M = _M(m, J)
        Sz[m,m] = M
    return Sz
        
# ==================================================================================== #
# |                            Declared Functions                                    | #
# ==================================================================================== #


def S_mats(N:int) -> Tuple[ np.matrix, np.matrix, np.matrix ] :
    # Check input:
    _assert_N(N)
    # Prepare Base Matrices:
    D_plus, D_minus = _D_plus_minus_mats(N)
    # Derive X, Y, Z Matrices
    Sx = D_plus + D_minus 
    Sy = -1j*D_plus + 1j*D_minus 
    Sz = _Sz_mat(N)
    # Return:
    return Sx, Sy, Sz

# ==================================================================================== #
# |                                  main                                            | #
# ==================================================================================== #


#TODO: 
#   M is from -J to J   
#   but we use the index 
#   m from 0 to N
""" _summary_
S_+ = ( 0   1 )
      ( 0   0 )

S_- = ( 0   0 )
      ( 1   0 )

S_X = ( 0   1 )
      ( 1   0 )

S_Z = ( 1   0 )
      ( 0  -1 )
"""

def _test_s_mats():
    _print = np_utils.print_mat
    for N in [2,4]:
        Sx, Sy, Sz = S_mats(N)
        print(f"N={N}")
        print("\nSx:")
        _print(Sx)
        print("\nSy:")
        _print(Sy)
        print("\nSz:")
        _print(Sz)
        print( "\n\n" )

def _test_M_of_m():
    # Init:
    N = 6
    M_vec = []
    m_vec = []
    # Compute:
    J = _J(N)
    for m in range(N+1):
        M = _M(m,J)
        M_vec.append(M)
        m_vec.append(m)
    # plot:
    plt.plot(m_vec, M_vec)
    plt.title(f"N={N}, J={J}")
    plt.grid(True)
    plt.xlabel("m")
    plt.ylabel("M")
    plt.show()

if __name__ == "__main__":
    np_utils.fix_print_length()
    # _test_M_of_m()
    _test_s_mats()
    print("Done.")