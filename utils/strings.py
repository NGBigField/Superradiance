# ==================================================================================== #
#|                                    Imports                                         |#
# ==================================================================================== #
if __name__ == "__main__":
    import pathlib, sys
    sys.path.append( str( pathlib.Path(__file__).parent.parent ) )

from typing import Any, Literal, Optional, Generator, TextIO, List

from utils import decorators, arguments, lists

import time
import string 

# for basic OOP:
from enum import Enum

# For defining print std_out or other:
import sys

# For manipulating methods of existing objects
import types


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
        self.printed_lines_lengths : List[int] = []
        self.print_out : TextIO = print_out
        self.in_place : bool = in_place

    @property
    def end_char(self)->str:
        if self.in_place:
            return ''
        else:
            return '\n'

    def _print(self, s:str, end:Optional[str]=None)->None:
        end = arguments.default_value(end, default=self.end_char)
        file = self.print_out
        print(s, end=end, file=file)
    
    @decorators.ignore_first_method_call
    def clear(self) -> None:
        # Get info about what was printed until now:
        reversed_prined_lengths = self.printed_lines_lengths.copy()
        reversed_prined_lengths.reverse()

        # Act according to `in_place`:
        for is_first, is_last, line_width in lists.iterate_with_edge_indicators(reversed_prined_lengths):
            if self.in_place:
                if not is_first:
                    pass   #TODO: Here we have a small bug that causes stacked static printers to override one-another                    
                self._print(SpecialChars.BackSpace*line_width)
                self._print(" "*line_width)
                self._print(SpecialChars.BackSpace*line_width)
                if not is_last:
                    self._print(SpecialChars.LineUp)
            else:
                self._print(SpecialChars.LineUp, end=SpecialChars.LineClear)

        # Reset printed lengths:
        self.printed_lines_lengths = []
                
    def print(self, s:str) -> None:
        self.clear()
        print_lines = s.split(SpecialChars.NewLine)
        self.printed_lines_lengths = [len(line) for line in print_lines]
        self._print(s)
    


class StaticNumOutOfNum():
    def __init__(self, expected_end:int, print_prefix:str="", print_suffix:str="", print_out:TextIO=sys.stdout, in_place:bool=False) -> None:
        self.static_printer : StaticPrinter = StaticPrinter(print_out=print_out, in_place=in_place)
        self.expected_end :int = expected_end    
        self.print_prefix :str = print_prefix
        self.print_suffix :str = print_suffix
        self.counter = 0
        self._as_iterator : bool = False
        # First print:
        if expected_end>0:
            self._show()

    def __next__(self) -> int:
        try:
            val = self.next()
        except StopIteration:
            self.clear()
            raise StopIteration
        return val

    def __iter__(self) -> "StaticNumOutOfNum":
        self._as_iterator = True
        return self

    def _check_end_iterations(self)->bool:
        return self._as_iterator and self.counter > self.expected_end

    def next(self, increment:int=1, extra_str:Optional[str]=None) -> int:
        self.counter += increment
        self._show(extra_str)
        if self._check_end_iterations():
            raise StopIteration
        return self.counter

    def append_extra_str(self, extra_str:str)->None:
        self._show(extra_str)

    def clear(self):
        self.static_printer.clear()

    def _print(self, s:str):
        self.static_printer.print( s + self.print_suffix )

    def _show(self, extra_str:Optional[str]=None):
        i = self.counter
        expected_end = int( self.expected_end )
        s = num_out_of_num(i, expected_end)
        self._print( s )


class ProgressBar(StaticNumOutOfNum):
    def __init__(self, expected_end:int, print_prefix:str="", print_suffix:str="", print_length:int=60, print_out:TextIO=sys.stdout, in_place:bool=False): 
        # Save basic data:        
        self.print_length :int = print_length
        super().__init__(expected_end, print_prefix, print_suffix, print_out, in_place)

    @staticmethod
    def inactive()->"ProgressBar":
        inactive_prog_bar : ProgressBar = type('obj', (object,), {"next":0, "clear":0})  # type: ignore		
        inactive_prog_bar.next = lambda increment=0, extra_str="s": 0
        inactive_prog_bar.clear = lambda: None
        return inactive_prog_bar

    @staticmethod
    def unlimited(print_prefix:str="", print_suffix:str="", print_length:int=60, print_out:TextIO=sys.stdout, in_place:bool=False)->"ProgressBar":
        return UnlimitedProgressBar(print_prefix=print_prefix, print_suffix=print_suffix, print_length=print_length, print_out=print_out, in_place=in_place)

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
        self._print(s)



class UnlimitedProgressBar(ProgressBar):
    def __init__(self, print_prefix: str = "", print_suffix: str = "", print_length: int = 60, print_out: TextIO = sys.stdout, in_place: bool = False):
        expected_end = -1
        super().__init__(expected_end, print_prefix, print_suffix, print_length, print_out, in_place)

    def _check_end_iterations(self)->Literal[False]:
        return False
        
    def _show(self, extra_str:Optional[str]=None):
        # Unpack properties:
        prefix = self.print_prefix
        print_length = int( self.print_length )
        marker_loc =self.counter % print_length
        if marker_loc == 0:
            marker_loc = print_length

        # Derive print:        
        s =  f"{prefix}["
        s += f"{('.'*(marker_loc-1))}"
        s += f"{u'█'}"
        s += f"{('.'*(print_length-marker_loc))}"
        s += f"] {self.counter:d}"

        if extra_str is not None:
            s += " "+extra_str

        # Print:
        self._print(s)

class PrintColors(Enum):
    DEFAULT = '\033[0m'
    # Styles
    BOLD = '\033[1m'
    ITALIC = '\033[3m'
    UNDERLINE = '\033[4m'
    UNDERLINE_THICK = '\033[21m'
    HIGHLIGHTED = '\033[7m'
    HIGHLIGHTED_BLACK = '\033[40m'
    HIGHLIGHTED_RED = '\033[41m'
    HIGHLIGHTED_GREEN = '\033[42m'
    HIGHLIGHTED_YELLOW = '\033[43m'
    HIGHLIGHTED_BLUE = '\033[44m'
    HIGHLIGHTED_PURPLE = '\033[45m'
    HIGHLIGHTED_CYAN = '\033[46m'
    HIGHLIGHTED_GREY = '\033[47m'

    HIGHLIGHTED_GREY_LIGHT = '\033[100m'
    HIGHLIGHTED_RED_LIGHT = '\033[101m'
    HIGHLIGHTED_GREEN_LIGHT = '\033[102m'
    HIGHLIGHTED_YELLOW_LIGHT = '\033[103m'
    HIGHLIGHTED_BLUE_LIGHT = '\033[104m'
    HIGHLIGHTED_PURPLE_LIGHT = '\033[105m'
    HIGHLIGHTED_CYAN_LIGHT = '\033[106m'
    HIGHLIGHTED_WHITE_LIGHT = '\033[107m'

    STRIKE_THROUGH = '\033[9m'
    MARGIN_1 = '\033[51m'
    MARGIN_2 = '\033[52m' # seems equal to MARGIN_1
    # colors
    BLACK = '\033[30m'
    RED_DARK = '\033[31m'
    GREEN_DARK = '\033[32m'
    YELLOW_DARK = '\033[33m'
    BLUE_DARK = '\033[34m'
    PURPLE_DARK = '\033[35m'
    CYAN_DARK = '\033[36m'
    GREY_DARK = '\033[37m'

    BLACK_LIGHT = '\033[90m'
    RED = '\033[91m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    PURPLE = '\033[95m'
    CYAN = '\033[96m'
    WHITE = '\033[96m'



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
    width = arguments.default_value(width, len(f"{val}"))
    format += f"{width}"            
    
    precision = arguments.default_value(precision, 0)
    format += f".{precision}f"    
        
    return f"{val:{format}}"  

def num_out_of_num(num1, num2):
    width = len(str(num2))
    format = lambda num: formatted(num, fill=' ', alignment='>', width=width )
    return format(num1)+"/"+format(num2)

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
def alphabet(upper_case:bool=False)->Generator[str, None, None]:
    if upper_case is True:
        l = list( string.ascii_uppercase )
    else:
        l = list( string.ascii_lowercase )
    for s in l:
        yield s

def add_color(s:str, color:PrintColors)->str:
    return f"{color} {s} {PrintColors.DEFAULT}"

def print_warning(*args, **kwargs)->None:
    warn1color = PrintColors.YELLOW_DARK
    warn2color = PrintColors.HIGHLIGHTED_YELLOW
    default = PrintColors.DEFAULT
    print(warn1color, "Warning: ", warn2color, *args, default, **kwargs)





# ==================================================================================== #
#|                                    main tests                                      |#
# ==================================================================================== #

def _progress_bar_example():    
    
    N = 3
    M = 4
    K = 50

    prog_bar = ProgressBar(N, "Big thing..... ", in_place=False)
    for i in range(N):        
        prog_bar.next()

        for j in range(M):        
            prog_bar.append_extra_str(f"Hello i=={i}, j=={j}")    

            prog_bar2 = ProgressBar.unlimited("Small thing... ")
            for k in prog_bar2:
                time.sleep(0.05)
                if k>K:
                    break
        
    prog_bar.clear()
    print("Done.")

def _printed_color_example():
    s = " Hello " + add_color("World", PrintColors.HIGHLIGHTED_PURPLE_LIGHT)
    print(s)


if __name__ == "__main__":
    _progress_bar_example()
    # _printed_color_example()