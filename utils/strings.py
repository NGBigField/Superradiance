# ==================================================================================== #
#|                                    Imports                                         |#
# ==================================================================================== #


import time 

from typing import (
    Optional,
    Literal,
    Any,
)


# ==================================================================================== #
#|                              declared functions                                    |#
# ==================================================================================== #

def formatted(s:Any, fill:str=' ', alignment:Literal['<','^','>']='>', width:Optional[int]=None, decimals:Optional[int]=None) -> str:
    if width is None:
        width = len(f"{s}")
    if decimals is None:
        s_out = f"{s:{fill}{alignment}{width}}"  
    else:
        s_out = f"{s:{fill}{alignment}{width}.{decimals}f}"  
    return s_out

def num_out_of_num(num1, num2):
    width = len(str(num2))
    formatted = lambda num: formatted(num, fill=' ', alignment='>', width=width )
    return formatted(num1)+"/"+formatted(num2)

def time_stamp():
    t = time.localtime()
    return f"{t.tm_year}.{t.tm_mon:02}.{t.tm_mday:02}_{t.tm_hour:02}.{t.tm_min:02}.{t.tm_sec:02}"

def insert_spaces_in_newlines(s:str, num_spaces:int) -> str:
    spaces = ' '*num_spaces
    s2 = s.replace('\n','\n'+spaces)
    return s2

def str_width(s:str, last_line_only:bool=False) -> int:
    lines = s.split('\n')
    widths = [len(line) for line in lines]
    if last_line_only:
        return widths[-1]
    else:
        return max(widths)
        

# ==================================================================================== #
#|                                      tests                                         |#
# ==================================================================================== #

def _main_test():
    import numpy as np
    mat = np.matrix([
        [1, 0, 0],
        [0, 0, 0],
        [0, 0, 0]
    ])
    s = insert_spaces_in_newlines(str(mat), 5)
    print(s)
    print(str_width(s))

if __name__ == "__main__":
    _main_test()