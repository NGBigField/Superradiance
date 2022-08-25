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


# ==================================================================================== #
# |                            Inner Functions                                       | #
# ==================================================================================== #

def _assert_N(N:int) -> None:
    assert float(N)/2 == int(int(N)/2) # N must be even    

def _J(N:int) -> int : 
    _assert_N(N)
    return N//2

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

def _d_plus_or_minus_mat(N:int, pm:int) -> np.matrix :    
    D = np.zeros(_mat_size(N))
    J = _J(N)
    for M in range(0, N+1):
        print(M)
        i = M
        j = M + pm
        if j>=N+1 or j<0:
            continue
        d = _d_plus_minus(M, J, pm)
        D[i,j] = d
    return D


# ==================================================================================== #
# |                            Declared Functions                                    | #
# ==================================================================================== #

def d_plus_mat(N:int) -> np.matrix:
    return _d_plus_or_minus_mat(N, +1)

def d_minus_mat(N:int) -> np.matrix:
    return _d_plus_or_minus_mat(N, -1)


def sx_mat(N:int) -> np.matrix:
    _assert_N(N)
    d_plus = d_plus_mat(N)
    d_minus = d_minus_mat(N)
    sx = d_plus+d_minus
    return sx



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

def _sx_test():
    N = 2
    sx = sx_mat(N)
    print(sx)

if __name__ == "__main__":
    _sx_test()