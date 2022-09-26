import time 

# For type hints:
from typing import (
    TypeVar,
)


# Matlab engine
try:
    import matlab.engine
except ImportError:
    _matlab_engine_on = False
else:
    _matlab_engine_on = True


# For type hints:
if _matlab_engine_on:
    MatlabEngineType = matlab.engine.matlabengine.MatlabEngine
else:
    MatlabEngineType = TypeVar('MatlabEngineType')



def init(add_sub_paths:bool=True) -> MatlabEngineType:
    eng = matlab.engine.start_matlab()
    if add_sub_paths:
        matlab.add_all_sub_paths(eng)
    return eng

def example() -> None:
    eng = matlab.init()
    eng.life(nargout=0)
    time.sleep(1)

def add_all_sub_paths(eng:MatlabEngineType) -> None:
    eng.addpath(eng.genpath(eng.pwd()),nargout=0)




# ============================================================================ #
#                                    Tests                                     #
# ============================================================================ #


if __name__ == "__main__":
    # Run Example Codes:    
    # Matlab.example()

    mat = matlab.init()

    print("a is prime?")
    print( mat.isprime(3) )
    res = mat.my_test(3)
    print(res)
    print("Done.")