# ============================================================================ #
#                                  Imports                                     #
# ============================================================================ #
import time 

# For type hints:
import typing as typ
from typing import (
    Optional,
    Any,
    Tuple,
    List,    
    TypeVar,
)

# Operating System and files:
from pathlib import Path
import os

from collections.abc import Iterable

# Matlab engine
try:
    import matlab.engine
except ImportError:
    _matlab_engine_on = False
else:
    _matlab_engine_on = True

# For controlling numpy behavior:
import numpy as np

# Visuals:
import matplotlib.pyplot as plt
from matplotlib.figure import Figure as FigureType


# ============================================================================ #
#                            Constants and Flags                               #
# ============================================================================ #

# For type hints:
if _matlab_engine_on:
    MatlabEngineType = matlab.engine.matlabengine.MatlabEngine
else:
    MatlabEngineType = TypeVar('MatlabEngineType')

# ============================================================================ #
#                               Static Classes                                 #
# ============================================================================ #

# ============================= visuals ====================================== #
class visuals:

    @staticmethod
    def save_figure(fig:Optional[FigureType]=None, file_name:Optional[str]=None ) -> None:
        # Figure:
        if fig is None:
            fig = plt.gcf()
        # Title:
        if file_name is None:
            file_name = strings.time_stamp()
        # Figures folder:
        folder = fullpath = Path().cwd().joinpath('figures')
        if not folder.is_dir():
            os.mkdir(str(folder.resolve()))
        # Full path:
        fullpath = folder.joinpath(file_name)
        fullpath_str = str(fullpath.resolve())+".png"
        # Save:
        fig.savefig(fullpath_str)
        return 

# ============================= matlab ====================================== #
class matlab():

    @staticmethod
    def init(add_sub_paths:bool=True) -> MatlabEngineType:
        eng = matlab.engine.start_matlab()
        if add_sub_paths:
            matlab.add_all_sub_paths(eng)
        return eng

    @staticmethod
    def example() -> None:
        eng = matlab.init()
        eng.life(nargout=0)
        time.sleep(1)

    @staticmethod
    def add_all_sub_paths(eng:MatlabEngineType) -> None:
        eng.addpath(eng.genpath(eng.pwd()),nargout=0)

# ============================= decorators ====================================== #
class decorators():

    @staticmethod
    def timeit(func: typ.Callable):
        def wrapper(*args, **kwargs):
            # Parse Input:
            funcName = func.__name__
            # Run:
            startingTime = time.time()
            results = func(*args, **kwargs)
            finishTime = time.time()
            # Print:
            print(f"Function '{funcName}' took {finishTime-startingTime} sec.")
            # return:
            return results            
        return wrapper

    @staticmethod
    def assert_type(Type: type, at: typ.Literal['input', 'output', 'both'] = 'both' ) -> typ.Callable:
        """assertType 

        Args:
            Type (type): type needed at input\output of function.
            at (InputOutput, optional): Defaults to InputOutput.Both.
        """
        # Parse options:
        if at == 'input':
            atInput  = True 
            atOutput = False
        elif at == 'output':
            atInput  = False 
            atOutput = True
        elif at == "both":
            atInput  = True 
            atOutput = True
        else:
            raise ValueError(f"Impossible Option")
        def decorator(func: typ.Callable):
            def wrapper(*args, **kwargs):
                # Define Assertion including error message:
                def _Assertion(arg):
                    assert isinstance(arg, Type), f"Assertion failed. Argument {arg} is not of type '{Type.__name__}'"
                # Assert Inputs :
                if atInput:
                    for arg in args:
                        _Assertion(arg)
                    for key, val in kwargs.items():
                        _Assertion(val)
                # Call func:
                results = func(*args, **kwargs)
                # Assert Outputs:
                if atOutput:
                    if isinstance(results, Iterable):
                        for result in results:
                            _Assertion(result)
                    else:
                        _Assertion(results)
                return results           
            return wrapper
        return decorator


# ============================= strings ====================================== #
class strings():
    def formatted(s:typ.Any, fill:str=' ', alignment:typ.Literal['<','^','>']='>', width:typ.Optional[int]=None, decimals:typ.Optional[int]=None) -> str:
        if width is None:
            width = len(f"{s}")
        if decimals is None:
            s_out = f"{s:{fill}{alignment}{width}}"  
        else:
            s_out = f"{s:{fill}{alignment}{width}.{decimals}f}"  
        return s_out
    
    def num_out_of_num(num1, num2):
        width = len(str(num2))
        formatted = lambda num: strings.formatted(num, fill=' ', alignment='>', width=width )
        return formatted(num1)+"/"+formatted(num2)

    def time_stamp():
        t = time.localtime()
        return f"{t.tm_year}.{t.tm_mon:02}.{t.tm_mday:02}_{t.tm_hour:02}.{t.tm_min:02}.{t.tm_sec:02}"


# ============================= assertions ====================================== #
class assertions():
    @staticmethod
    def integer(x:Any)->None:
        assert round(x) == x

    @staticmethod
    def even(x:Any)->None:
        assertions.integer(x)
        assert float(x)/2 == int(int(x)/2)

# ============================= numpy ====================================== #
class numpy:
    @staticmethod
    def fix_print_length(linewidth:int=10000, precision:int=2) -> None:
        np.set_printoptions(linewidth=linewidth)
        np.set_printoptions(precision=precision)

    @staticmethod
    def print_mat(m:np.matrix) -> None:
        if np.all( np.isreal(m*1j) ):  # if all matrix is imaginary
            real_mat = np.real( m*1j )
            print(f"{real_mat} * 1j")
        else:
            print(m)





# ============================================================================ #
#                                    Tests                                     #
# ============================================================================ #

@decorators.assert_type(int, at='output')
@decorators.assert_type(str, at='input' )
def _assertType_Example(a, b, c='Hello', d='Bye'):    
    print(f"a='{a}' , b='{b}' , c='{c}' , d='{d}'")
    res = 3
    print(f"res={res}")
    return res

if __name__ == "__main__":
    # Run Example Codes:    
    # Matlab.example()

    mat = matlab.init()

    print("a is prime?")
    print( mat.isprime(3) )
    res = mat.my_test(3)
    print(res)
    print("Done.")


