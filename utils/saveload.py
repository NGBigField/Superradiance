# ==================================================================================== #
#|                                    Imports                                         |#
# ==================================================================================== #

# Typing hints:
from typing import (
    Tuple,
    Optional,
    Any,
    List,
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
DATA_FOLDER = os.getcwd()+os.sep+"saved_data"+os.sep

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
def _open(name:str, mode:str):
    folder = DATA_FOLDER
    make_sure_folder_exists(folder)
    full_path = folder+name
    return open(full_path, mode)

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

def save(var:Any, name:Optional[str]=None) -> None:
    # Complete missing inputs:
    name = args.default_value(name, strings.time_stamp())
    # Prepare pickle inputs:
    name = _common_name(name)
    mode = Mode.Write.str()
    file = _open(name, mode)
    # Save:
    pickle.dump(var, file)

def load(name:str) -> Any:
    name = _common_name(name)
    mode = Mode.Read.str()
    file = _open(name, mode)
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

def make_sure_folder_exists(foldepath:str) -> None:
    if not os.path.exists(foldepath):
        os.makedirs(foldepath)


# ==================================================================================== #
#|                                      Tests                                         |#
# ==================================================================================== #

def _test():
    d = dict(a="A", b=3.05)
    save(d, "test_file")
    del d
    e = load("test_file.dat")
    print(e)


if __name__ == "__main__":
    _test()
    print("Done.")