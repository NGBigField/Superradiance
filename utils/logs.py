# ============================================================================ #
#|                                  Imports                                   |#
# ============================================================================ #
import sys

if __name__ == "__main__":
    import pathlib
    sys.path.append(pathlib.Path(__file__).parent.parent.__str__())
       
import logging
from typing import Final
from utils import strings, saveload
from enum import Enum, auto
import os


# ============================================================================ #
#|                                  Helpers                                   |#
# ============================================================================ #   

# Logger config:
DEFAULT_PRINT_FORMAT : Final = "%(asctime)s %(levelname)s %(message)s"
DEFAULT_DATE_TIME_FORMAT : Final = "%Y-%m-%d %H:%M:%S"

# File system:
PATH_SEP = os.sep
LOGS_FOLDER = os.getcwd()+PATH_SEP+"logs"


class LoggerLevels(Enum):
    DEBUG     = logging.DEBUG
    INFO      = logging.INFO
    WARNING   = logging.WARNING
    ERROR     = logging.ERROR
    CRITICAL  = logging.CRITICAL
    Lowest    = logging.DEBUG
    

# ============================================================================ #
#|                                 Constants                                  |#
# ============================================================================ #   



# ============================================================================ #
#|                               Inner Functions                              |#
# ============================================================================ #   

def _force_log_file_extension(filename:str)->str:
    file_parts = filename.split(".")
    extension = file_parts[-1]
    if extension != "log":
        filename += ".log"
    return filename

def _get_fullpath(filename:str)->str:
    saveload.force_folder_exists(LOGS_FOLDER)
    fullpath = LOGS_FOLDER+PATH_SEP+filename
    return fullpath

# ============================================================================ #
#|                               Inner Classes                                |#
# ============================================================================ #           

class _MYFormatter(logging.Formatter):

    def format(self, record)->str:
        s = super().format(record)
        level_value = record.levelno

        if level_value in [LoggerLevels.CRITICAL.value, LoggerLevels.ERROR.value]:
            color = strings.PrintColors.RED
            s = strings.add_color(s, color)
        elif level_value == LoggerLevels.WARNING.value:
            warn1color = strings.PrintColors.YELLOW_DARK
            warn2color = strings.PrintColors.HIGHLIGHTED_YELLOW
            s = strings.add_color("Warning:", warn1color) + strings.add_color(s, warn2color)

        return s
        
# ============================================================================ #
#|                             Declared Functions                             |#
# ============================================================================ #           

def get_logger(
    level:LoggerLevels=LoggerLevels.INFO,
    filename:str=strings.time_stamp(),
    name:str="logger"
    )->logging.Logger:
    
    # Get logger obj:
    logger = logging.getLogger(name)
    logger.propagate = False
    logger.setLevel(LoggerLevels.Lowest.value)
    
    # Derive fullpath:
    filename = _force_log_file_extension(filename)
    fullpath = _get_fullpath(filename)
    
    ## Configuration:
    f_formatter = logging.Formatter(fmt=DEFAULT_PRINT_FORMAT, datefmt=DEFAULT_DATE_TIME_FORMAT)
    c_formatter = _MYFormatter(fmt="%(message)s")
    #
    f_handler = logging.FileHandler(fullpath)
    c_handler = logging.StreamHandler(sys.stdout)
    #
    f_handler.setFormatter(f_formatter)
    c_handler.setFormatter(c_formatter)
    #
    f_handler.setLevel(logging.DEBUG)  # Write all logs to file
    c_handler.setLevel(level.value)    # Print only logs above level
    
    ## set handlers:        
    logger.addHandler(f_handler)      
    logger.addHandler(c_handler)      
    
    return logger



# ============================================================================ #
#|                                    Test                                    |#
# ============================================================================ #     


def _main_test():
    logger = get_logger()
    logger.debug("1. debug: Bug?")
    logger.info("2. info: Hello...")
    logger.warning("3. warning: Warn Somebody")
    logger.error("4. error: Failing!")
    logger.critical("5. critical: Help!!!")

if __name__ == "__main__":
    _main_test()