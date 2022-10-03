import time 

from typing import (
    Optional,
    Literal,
    Any,
)

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