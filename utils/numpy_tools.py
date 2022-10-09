import numpy as np
EPS = 1e-14

def fix_print_length(linewidth:int=10000, precision:int=2) -> None:
    np.set_printoptions(linewidth=linewidth)
    np.set_printoptions(precision=precision)

def _reduce(val):
    if abs(val)<EPS:
        return 0
    else:
        return val

def mat_str(mat:np.matrix) -> str:
    # check:
    assert isinstance(mat, np.ndarray)
    m = mat.copy()
    # reduce values close to zero:
    for idx, x in np.ndenumerate(m):
        m[idx] = _reduce(np.real(x)) + 1j*_reduce(np.imag(x))
    # print:
    if np.all( np.isreal(m) ):  # if all matrix is real
        return f"{np.real(m)}"
    elif np.all( np.isreal(m*1j) ):  # if all matrix is imaginary
        return f"{np.imag(m)} * 1j"
    else:
        return f"{m}"

def print_mat(m:np.matrix) -> None:
    print(mat_str(m))