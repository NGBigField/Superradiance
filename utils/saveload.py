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
    Literal,
)

from numpy import isin

# Other utilities:
try:
    from utils import (
        strings,
        errors,
        arguments,
        assertions,
    )
except ImportError:
    import strings
    import errors
    import arguments
    import assertions

# Operating System and files:
from pathlib import Path
import os

# For saving stuff:
import pickle
import csv

# ==================================================================================== #
#|                                  Constants                                         |#
# ==================================================================================== #
PATH_SEP = os.sep
DATA_FOLDER = os.getcwd()+PATH_SEP+"data"
DATA_EXTENSION = "dat"
LOG_EXTENSION = "log"

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

    class Append():
        @classmethod
        def str(cls) -> str:
            return 'a'
# ==================================================================================== #
#|                               Inner Functions                                      |#
# ==================================================================================== #
def _fullpath(name:str, sub_folder:Optional[str]=None, typ:Literal['data', 'log']='data') -> str:
    # Complete missing inputs:
    sub_folder = arguments.default_value(sub_folder, "")
    name = _common_name(name, typ)
    folder = DATA_FOLDER+PATH_SEP+sub_folder
    force_folder_exists(folder)
    fullpath = folder+PATH_SEP+name
    return fullpath
    
def _open(fullpath:str, mode:str):
    return open(fullpath, mode)

def _common_name(name:str, typ:Literal['data', 'log']='data') -> str:
    assert isinstance(name, str)

    given_extension = name[-4:]
    if typ=="data":
        target_extension = DATA_EXTENSION
    elif typ=="log":
        target_extension = LOG_EXTENSION
    else:
        raise ValueError(f"Not a valid `typ` input. Given '{typ}'")
        
    if given_extension == "."+target_extension:
        return name
    else:
        return name+"."+target_extension


# ==================================================================================== #
#|                              Declared Functions                                    |#
# ==================================================================================== #
def append_text(text:str, name:str, sub_folder:Optional[str]=None, in_new_line:bool=True) -> None:
    assert isinstance(text, str), f"`text` Must be of type str"
    if in_new_line:
        text = "\n"+text
    fullpath = _fullpath(name, sub_folder, typ='log')
    mode = Mode.Append.str()
    flog = open(fullpath, mode)
    flog.write(text)
    flog.close()

def exist(name:str, sub_folder:Optional[str]=None) -> bool:
    fullpath = _fullpath(name, sub_folder)
    return os.path.exists(fullpath)

def all_saved_data() -> Generator[Tuple[str, Any], None, None]:
    for path, subdirs, files in os.walk(DATA_FOLDER):
        for name in files:
            fullpath = path + PATH_SEP + name
            file = _open(fullpath, Mode.Read.str())
            data = pickle.load(file)
            yield name, data

def save(var:Any, name:Optional[str]=None, sub_folder:Optional[str]=None) -> None:
    # Complete missing inputs:
    name = arguments.default_value(name, strings.time_stamp())
    # fullpath:
    fullpath = _fullpath(name, sub_folder)
    # Prepare pickle inputs:
    mode = Mode.Write.str()
    file = _open(fullpath, mode)
    # Save:
    pickle.dump(var, file)

def load(name:str, sub_folder:Optional[str]=None) -> Any:
    # fullpath:
    fullpath = _fullpath(name, sub_folder)
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

def force_subfolder_exists(folder_name:str) -> None:
    folderpath = DATA_FOLDER + PATH_SEP + folder_name
    force_folder_exists(folderpath)

def force_folder_exists(folderpath:str) -> None:
    if not os.path.exists(folderpath):
        os.makedirs(folderpath)


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