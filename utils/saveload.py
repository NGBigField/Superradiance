# ==================================================================================== #
#|                                    Imports                                         |#
# ==================================================================================== #

# Typing hints:
from typing import (
    Tuple,
    Optional,
    Any,
    List,
    Generator,
)

from numpy import isin

# Other utilities:
try:
    from utils import (
        strings,
        errors,
        args,
    )
except ImportError:
    import strings
    import errors
    import args

# Operating System and files:
from pathlib import Path
import os

# For saving stuff:
import pickle
import csv

# ==================================================================================== #
#|                                  Constants                                         |#
# ==================================================================================== #
DATA_FOLDER = os.getcwd()+os.sep+"_saved_data"+os.sep
PATH_SEP = os.sep

# ==================================================================================== #
#|                                   Classes                                          |#
# ==================================================================================== #

class Mode():     
    class Read():
        @classmethod
        def str(cls) -> str:
            return 'rb'
    class Write():
        @classmethod
        def str(cls) -> str:
            return 'wb'

# ==================================================================================== #
#|                               Inner Functions                                      |#
# ==================================================================================== #
def _fullpath(name:str) -> str:
    name = _common_name(name)
    folder = DATA_FOLDER
    force_folder_exists(folder)
    fullpath = folder+name
    return fullpath
    
def _open(fullpath:str, mode:str):
    return open(fullpath, mode)

def _common_name(name:str) -> str:
    assert isinstance(name, str)
    extension = name[-4:]
    if extension == ".dat":
        return name
    else:
        return name+".dat"


# ==================================================================================== #
#|                              Declared Functions                                    |#
# ==================================================================================== #

def all_saved_data() -> Generator[Tuple[str, Any], None, None]:
    for path, subdirs, files in os.walk(DATA_FOLDER):
        for name in files:
            fullpath = path + PATH_SEP + name
            file = _open(fullpath, Mode.Read.str())
            data = pickle.load(file)
            yield name, data

def save(var:Any, name:Optional[str]=None) -> None:
    # Complete missing inputs:
    name = args.default_value(name, strings.time_stamp())
    # fullpath:
    fullpath = _fullpath(name)
    # Prepare pickle inputs:
    mode = Mode.Write.str()
    file = _open(fullpath, mode)
    # Save:
    pickle.dump(var, file)

def load(name:str) -> Any:
    # fullpath:
    fullpath = _fullpath(name)
    # Prepare pickle inputs:
    mode = Mode.Read.str()
    file = _open(fullpath, mode)
    return pickle.load(file)

def save_table(table:List[List[str]], filename:Optional[str]=None) -> None :
    # Complete missing inputs:
    if filename is None:
        filename = strings.time_stamp()

    try:
        with open(filename+".csv", 'w') as f:        
            write = csv.writer(f)        
            write.writerows(table)
    except Exception as e:
        errors.print_traceback(e)
    finally:
        save(table, name=filename)

def force_folder_exists(foldepath:str) -> None:
    if not os.path.exists(foldepath):
        os.makedirs(foldepath)


# ==================================================================================== #
#|                                      Tests                                         |#
# ==================================================================================== #

def _test():
    all_saved_data()


if __name__ == "__main__":
    _test()
    print("Done.")