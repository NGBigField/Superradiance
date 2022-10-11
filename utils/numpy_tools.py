# ==================================================================================== #
#|                                    Imports                                         |#
# ==================================================================================== #

import numpy as np
from typing import (
    Union,
)

try:
    from utils import(
        errors,
        types,
        strings,
    )
except ImportError:
    import sys, pathlib
    sys.path.append( 
            pathlib.Path(__file__).parent.parent.__str__()
    )
    from utils import(
        errors,
        types,
        strings,
    )
    


# ==================================================================================== #
#|                                   Constants                                        |#
# ==================================================================================== #
EPS = 1e-14

# ==================================================================================== #
#|                               inner functions                                      |# 
# ==================================================================================== #
def _reduce(val:Union[float, int]) -> Union[float, int] :
    if abs(val)<EPS:
        return 0
    else:
        return val

def _reduce_complex(val:Union[float, int, complex], data_type:type) -> Union[float, int, complex] :
    re = _reduce(np.real(val))
    im = _reduce(np.imag(val))
    if data_type is complex:
        return complex(re,im)
    else:
        return re
# ==================================================================================== #
#|                              declared functions                                    |#
# ==================================================================================== #
def fix_print_length(linewidth:int=10000, precision:int=4) -> None:
    np.set_printoptions(
        linewidth=linewidth,
        precision=precision
    )

def mat_str(mat:np.matrix) -> str:
    # check:
    assert isinstance(mat, np.ndarray)
    m = mat.copy()
    # reduce values close to zero:
    data_type = types.numpy_dtype_to_std_type(m.dtype)
    for idx, x in np.ndenumerate(m):
        m[idx] = _reduce_complex(x, data_type)
    # print:
    if np.all( np.isreal(m) ):  # if all matrix is real
        return f"{np.real(m)}"
    elif np.all( np.isreal(m*1j) ):  # if all matrix is imaginary
        return f"{np.imag(m)} * 1j"
    else:
        return f"{m}"

def mat_str_with_leading_text(mat:np.matrix, text:str) -> str:
    assert isinstance(text, str)
    text_width = strings.str_width(text, last_line_only=True)
    s = mat_str(mat)
    return text + strings.insert_spaces_in_newlines(s, num_spaces=text_width)

def print_mat(m:np.matrix) -> None:
    print(mat_str(m))

def print_mat_with_leading_text(mat:np.matrix, text:str) -> None:
    print(mat_str_with_leading_text(mat, text))

# ==================================================================================== #
#|                                      tests                                         |#
# ==================================================================================== #

def _main_test():
    name = 'm is a matrix = '
    mat = np.matrix([
        [1, 0, 0],
        [0, 0, 0],
        [0, 0, 0]
    ])
    s = mat_str_with_leading_text(1j*mat, name)
    print(s)

if __name__ == "__main__":
    _main_test()