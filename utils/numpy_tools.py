import numpy as np


def fix_print_length(linewidth:int=10000, precision:int=2) -> None:
    np.set_printoptions(linewidth=linewidth)
    np.set_printoptions(precision=precision)

def mat_str(m:np.matrix) -> str:
    if np.all( np.isreal(m*1j) ):  # if all matrix is imaginary
        real_mat = np.real( m*1j )
        return f"{real_mat} * 1j"
    else:
        return f"{m}"

def print_mat(m:np.matrix) -> None:
    print(mat_str(m))