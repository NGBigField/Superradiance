        
# ==================================================================================== #
# |                                   Imports                                        | #
# ==================================================================================== #

# The main utility that manages pickles
from utils import saveload

# other useful utilities:
from utils import assertions

# file system managers:
from pathlib import Path
import os

# for oop style:
from dataclasses import dataclass
        
# For typing hints:
from typing import(
    Any,    
    List,
)
from algo.coherentcontrol import Operation
import numpy as np

        
# ==================================================================================== #
# |                                  Constants                                       | #
# ==================================================================================== #

NOON_FOLDER_RELATIVE_PATH = "Noons"
NOON_FOLDER_FULLPATH = saveload.DATA_FOLDER+ NOON_FOLDER_RELATIVE_PATH


# ==================================================================================== #
# |                                   Classes                                        | #
# ==================================================================================== #

@dataclass
class NOON_DATA:
    num_moments     : int
    state           : np.ndarray
    operation       : List[Operation]
    params          : List[float]
    fidelity        : float

# ==================================================================================== #
# |                               Inner Functions                                    | #
# ==================================================================================== #

def _standard_relative_path(num_moments:int) -> str:
    # Check input:
    num_moments = assertions.integer(num_moments)
    # Return:
    return NOON_FOLDER_RELATIVE_PATH + os.sep + f"noon_{num_moments}"

def _standard_full_path(num_moments:int) -> str:
    # Check input:
    num_moments = assertions.integer(num_moments)
    # Check Folder:
    saveload.make_sure_folder_exists(NOON_FOLDER_FULLPATH)
    return NOON_FOLDER_FULLPATH + os.sep + f"noon_{num_moments}.dat"

# ==================================================================================== #
# |                              Declared Functions                                  | #
# ==================================================================================== #



def exist_saved_noon(num_moments:int) -> bool:
    fullpath = _standard_full_path(num_moments)
    return os.path.exists(fullpath)

def get_saved_noon(num_moments:int) -> NOON_DATA:
    assert exist_saved_noon(num_moments), f"Doesn't contain saved noon data with num_moments={num_moments}"
    relative_path = _standard_relative_path(num_moments)
    return saveload.load(relative_path)
    
def save_noon(data:NOON_DATA):
    assert isinstance(data, NOON_DATA)
    relative_path = _standard_relative_path(data.num_moments)
    saveload.save(data, relative_path)
    


# ==================================================================================== #
# |                                    Tests                                         | #
# ==================================================================================== #


def _test():
    print("test!")
    is_exist = exist_saved_noon(40)
    print(f"is_exist={is_exist}")
    

if __name__ == "__main__":
    _test()
    print("Done.")
