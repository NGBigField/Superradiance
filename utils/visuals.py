# Typing hints:
from typing import (
    Optional,
    Literal,
)

# Operating System and files:
from pathlib import Path
import os

# For plotting:
import matplotlib.pyplot as plt
from matplotlib.figure import Figure as FigureType


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