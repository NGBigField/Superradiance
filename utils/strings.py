# ==================================================================================== #
#|                                    Imports                                         |#
# ==================================================================================== #


import time 

from typing import (
    Optional,
    Literal,
    Any,
    TextIO,
    List,
)

from enum import StrEnum

# use other utilities:
try:
    from utils import (
        decorators,
        args,
        lists,
    )
except ImportError:
    import decorators
    import args
    import lists

# For defining print std_out or other:
import sys

# ==================================================================================== #
#|                                  Constants                                         |#
# ==================================================================================== #

class SpecialChars:
    NewLine = "\n"
    CarriageReturn = "\r"
    Tab = "\t"
    BackSpace = "\b"
    LineUp = '\033[1A'
    LineClear = '\x1b[2K'

# ==================================================================================== #
#|                               declared classes                                     |#
# ==================================================================================== #

class StaticPrinter():

    def __init__(self, print_out:TextIO=sys.stdout, in_place:bool=False) -> None:
        self.printed_lines_lengths : List[int] = [0]
        self.print_out : TextIO = print_out
        self.in_place : bool = in_place

    @property
    def end_char(self)->str:
        if self.in_place:
            return ''
        else:
            return '\n'

    def _print(self, s:str, end:Optional[str]=None)->None:
        end = args.default_value(end, default=self.end_char)
        file = self.print_out
        print(s, end=end, file=file)
    
    @decorators.ignore_first_method_call
    def clear(self) -> None:
        reversed_prined_lengths = self.printed_lines_lengths.copy()
        reversed_prined_lengths.reverse()
        for is_first, is_last, line_width in lists.iterate_with_edge_indicators(reversed_prined_lengths):
            if self.in_place:
                self._print(SpecialChars.BackSpace*line_width)
                self._print(" "*line_width)
                self._print(SpecialChars.BackSpace*line_width)
                if not is_last:
                    self._print(SpecialChars.LineUp)
            else:
                self._print(SpecialChars.LineUp, end=SpecialChars.LineClear)
                

    def print(self, s:str) -> None:
        self.clear()
        print_lines = s.split(SpecialChars.NewLine)
        self.printed_lines_lengths = [len(line) for line in print_lines]
        self._print(s)
    


class ProgressBar():
    def __init__(self, expected_end:int, print_prefix:str="", print_length:int=60, print_out:TextIO=sys.stdout): 
        self.static_printer : StaticPrinter = StaticPrinter(print_out=print_out, in_place=False)
        self.expected_end :int = expected_end
        self.print_prefix :str = print_prefix
        self.print_length :int = print_length
        self.counter = 0
        self._as_iterator : bool = False

    def __next__(self) -> int:
        return self.next()

    def __iter__(self):
        self._as_iterator = True
        return self

    def next(self, increment:int=1, extra_str:Optional[str]=None) -> int:
        self.counter += increment
        if self._as_iterator and self.counter > self.expected_end:
            self.clear()
            raise StopIteration
        self._show(extra_str)
        return self.counter

    def clear(self):
        self.static_printer.clear()

    def _show(self, extra_str:Optional[str]=None):
        # Unpack properties:
        i = self.counter
        prefix = self.print_prefix
        expected_end = int( self.expected_end )
        print_length = int( self.print_length )

        # Derive print:
        if i>expected_end:
            crnt_bar_length = print_length
        else:
            crnt_bar_length = int(print_length*i/expected_end)
        s = f"{prefix}[{u'█'*crnt_bar_length}{('.'*(print_length-crnt_bar_length))}] {i:d}/{expected_end:d}"

        if extra_str is not None:
            s += " "+extra_str

        # Print:
        self.static_printer.print(s)



# ==================================================================================== #
#|                              declared functions                                    |#
# ==================================================================================== #

def formatted(
    val:Any, 
    fill:str=' ', 
    alignment:Literal['<','^','>']='>', 
    width:Optional[int]=None, 
    precision:Optional[int]=None,
    signed:bool=False
) -> str:
    
    # Check info:
    try:
        if round(val)==val and precision is None:
            force_int = True
        else:
            force_int = False
    except:
        force_int = False
        
        
    
    # Simple formats:
    format = f"{fill}{alignment}"
    if signed:
        format += "+"
    
    # Width:
    width = args.default_value(width, len(f"{val}"))
    format += f"{width}"            
    
    
    precision = args.default_value(precision, 0)
    format += f".{precision}f"    
        
    s = f"{val:{format}}"  
    
    return s

def num_out_of_num(num1, num2):
    width = len(str(num2))
    formatted_ = lambda num: formatted(num, fill=' ', alignment='>', width=width )
    return formatted_(num1)+"/"+formatted_(num2)

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
        
def num_lines(s:str)->int:
    n = s.count(SpecialChars.NewLine)
    return n + 1


# ==================================================================================== #
#|                                      tests                                         |#
# ==================================================================================== #

def _main_test():

    x = 3.1415e-15
    print(formatted(x, width=8))

if __name__ == "__main__":
    _main_test()
