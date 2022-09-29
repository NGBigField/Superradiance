# ==================================================================================== #
# |                                   Imports                                        | #
# ==================================================================================== #

from typing import (
    Any,
    Union,
    Optional,
    TypeVar,
)

import numpy as np

# ==================================================================================== #
# |                                  Constants                                       | #
# ==================================================================================== #
EPSILON = 0.1  # Used for distance validation

# ==================================================================================== #
# |                                    Types                                         | #
# ==================================================================================== #

_T = TypeVar('_T')

# ==================================================================================== #
# |                               Inner Functions                                    | #
# ==================================================================================== #

def _assert(condition:bool, reason:Optional[str]=None, default_reason:Optional[str]=None):
    # if condition passed successfully:
    if condition:
        return
    # If error is needed:
    if reason is not None and isinstance(reason, str):
        raise AssertionError(reason)
    elif default_reason is not None and isinstance(default_reason, str):
        raise AssertionError(default_reason)
    else:
        raise AssertionError()


def _is_positive_semidefinite(m:np.matrix) -> bool:
    eigen_vals = np.linalg.eigvals(m)
    if np.any(np.imag(eigen_vals)>EPSILON):  # Must be real
        return False
    if np.any(np.real(eigen_vals)<-EPSILON):  # Must be positive
        return False
    return True

def _is_hermitian(m:np.matrix) -> bool:
    diff = m.H-m
    shape = m.shape
    for i in range(shape[0]):
        for j in range(shape[1]):
            if abs(diff[i,j])>EPSILON:
                return False
    return True  

# ==================================================================================== #
# |                              Declared Functions                                  | #
# ==================================================================================== #

def integer(x:_T, reason:Optional[str]=None) -> _T:
    _assert( isinstance(x, (int, float)), reason=reason )
    _assert( round(x) == x, reason=reason )
    return x

def index(x:_T, reason:Optional[str]=None) -> _T:
    integer(x, reason=reason)
    _assert( x >= 0, reason=reason )
    return x

def bit(x:_T, reason:Optional[str]=None) -> _T:
    integer(x, reason=reason)
    _assert( x in [0, 1], reason=reason )
    return x

def even(x:_T, reason:Optional[str]=None) -> _T:
    integer(x, reason=reason)
    _assert( float(x)/2 == int(int(x)/2), reason=reason )
    return x

def density_matrix(m:_T, reason:Optional[str]=None, robust_check:bool=True) -> _T:
    _assert( isinstance(m, (np.matrix, np.ndarray)), reason=reason, default_reason="Must be a matrix type" )
    if not isinstance(m, np.matrix):
        m = np.matrix(m)
    _assert( len(m.shape)==2, reason=reason, default_reason="Must be a matrix" )
    _assert( m.shape[0]==m.shape[1], reason=reason, default_reason="Must be a square matrix" )
    _assert( abs(np.trace(m)-1)<EPSILON, reason=reason, default_reason="Density Matrix must have trace==1")
    if robust_check:
        _assert( _is_hermitian(m), reason=reason, default_reason="Density Matrix must be hermitian")
        _assert( _is_positive_semidefinite(m), reason=reason, default_reason="Density Matrix must be positive semidefinite")
    return m