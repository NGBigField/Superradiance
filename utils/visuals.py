
# ==================================================================================== #
# |                                 Imports                                          | #
# ==================================================================================== #

# For plotting:
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.figure import Figure as FigureType
from matplotlib.cm import ScalarMappable
from qutip.matplotlib_utilities import complex_phase_cmap
from mpl_toolkits.mplot3d import Axes3D

# For defining print std_out or other:
import sys

# Everyone needs numpy in their life:
import numpy as np

# For type hints:
from typing import (
    Optional,
    Union,
)
from matplotlib.axes import Axes

# For function version detection:
from packaging.version import parse as parse_version

# Operating System and files:
from pathlib import Path
import os

# For plotting:
import matplotlib.pyplot as plt

# Import our tools and utils:
from utils import (
    strings,
)


# ==================================================================================== #
# |                             Inner Functions                                      | #
# ==================================================================================== #

if parse_version(mpl.__version__) >= parse_version('3.4'):
    def _axes3D(fig, *args, **kwargs):
        ax = Axes3D(fig, *args, auto_add_to_figure=False, **kwargs)
        return fig.add_axes(ax)
else:
    def _axes3D(*args, **kwargs):
        return Axes3D(*args, **kwargs)


# ==================================================================================== #
# |                            Declared Functions                                    | #
# ==================================================================================== #


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

def plot_city(M:Union[np.matrix, np.array], title:Optional[str]=None, ax:Axes=None):
    # Check input type:
    if isinstance(M, np.matrix):
        M = np.array(M)
    assert len(M.shape)==2
    assert M.shape[0]==M.shape[1]

    # Define common symbols:
    pi = np.pi

    n = np.size(M)
    xpos, ypos = np.meshgrid(range(M.shape[0]), range(M.shape[1]))
    xpos = xpos.T.flatten() - 0.5
    ypos = ypos.T.flatten() - 0.5
    zpos = np.zeros(n)
    dx = dy = 0.8 * np.ones(n)
    Mvec = M.flatten()
    dz = abs(Mvec).tolist()

    # make small numbers real, to avoid random colors
    idx = np.where(abs(Mvec) < 0.001)
    Mvec[idx] = abs(Mvec[idx])

    # Define colors:
    phase_min = -pi
    phase_max = pi
    norm = mpl.colors.Normalize(phase_min, phase_max)
    cmap = complex_phase_cmap()
    colors = cmap(norm(np.angle(Mvec)))[0]

    if ax is None:
        fig = plt.figure()
        ax = _axes3D(fig, azim=-35, elev=35)

    ax.bar3d(xpos, ypos, zpos, dx, dy, dz, color=colors)

    if title is not None:
        ax.set_title(title, y=0.95)

    # x axis
    xtics = -0.5 + np.arange(M.shape[0])
    ax.axes.w_xaxis.set_major_locator(plt.FixedLocator(xtics))
    ax.tick_params(axis='x', labelsize=12)

    # y axis
    ytics = -0.5 + np.arange(M.shape[1])
    ax.axes.w_yaxis.set_major_locator(plt.FixedLocator(ytics))
    ax.tick_params(axis='y', labelsize=12)

    # z axis
    ax.set_zlim3d([0, 1])  # use min/max

    # Labels:
    n = M.shape[0]
    plt.xticks( range(n), [ f"|{m}>" for m in range(n)] )
    plt.yticks( range(n), [ f"<{m}|" for m in range(n)] )

    # Colorbar:
    cax = plt.axes([0.90, 0.1, 0.05, 0.8])
    mappable = ScalarMappable(norm=norm, cmap=cmap)
    clr_bar = plt.colorbar(ax=ax, cax=cax, mappable=mappable )
    clr_bar.set_ticks([-pi, -pi/2, 0, pi/2, pi])
    clr_bar.set_ticklabels( [r'$-\pi$', r'$-\pi/2$', 0, r'$\pi/2$', r'$\pi$'] )
    
    return fig, ax

def plot_superradiance_evolution(times, energies, intensities):
    # Plot:
    fig, axes = plt.subplots(nrows=2, ncols=1, constrained_layout=True)

    axes[0].plot(times[0:-1], energies[0:-1])    
    axes[0].grid(which='major')
    axes[0].set_xlabel('Time [sec] '    , fontdict=dict(size=16) )
    axes[0].set_ylabel('Energy     '    , fontdict=dict(size=16) )
    axes[0].set_title('Evolution   '    , fontdict=dict(size=16) )

    axes[1].plot(times[0:-1], intensities)    
    axes[1].grid(which='major')
    axes[1].set_xlabel('Time [sec] '    , fontdict=dict(size=16) )
    axes[1].set_ylabel('Intensity  '    , fontdict=dict(size=16) )
    save_figure()
    plt.show()


class ProgressBar():

    def __init__(self, expected_end:int, print_prefix:str="", print_length:int=60, print_out=sys.stdout): # Python3.6+
        self.expected_end = expected_end
        self.print_prefix = print_prefix
        self.print_length = print_length
        self.print_out = print_out
        self.counter = 0
        self._as_iterator : bool = False

    def __next__(self) -> int:
        return self.next()

    def __iter__(self):
        self._as_iterator = True
        return self

    def next(self) -> int:
        self.counter += 1
        if self._as_iterator and self.counter > self.expected_end:
            self.close()
            raise StopIteration
        self._show()
        return self.counter

    def close(self):
        # print("", flush=True, file=self.print_out)
        full_bar_length = self.print_length+len(self.print_prefix)+4+len(str(self.expected_end))*2
        print(
            f"{(' '*(full_bar_length))}", 
            end='\r', 
            file=self.print_out, 
            flush=True
        )

    def _show(self):
        # Unpack properties:
        i = self.counter
        prefix = self.print_prefix
        expected_end = self.expected_end
        print_length = self.print_length

        # Derive print:
        if i>expected_end:
            crnt_bar_length = print_length
        else:
            crnt_bar_length = int(print_length*i/expected_end)
        s = f"{prefix}[{u'█'*crnt_bar_length}{('.'*(print_length-crnt_bar_length))}] {i}/{expected_end}"

        # Print:
        print(
            s,
            end='\r', 
            file=self.print_out, 
            flush=True
        )


# ==================================================================================== #
# |                                  Tests                                           | #
# ==================================================================================== #

def _test_prog_bar_as_iterator():
    import time    
    n = 5
    for p in ProgressBar(n, "Computing: "):
        time.sleep(0.1) # any code you need
    print("Done iteration")

def _test_prog_bar_as_object():
    import time    
    n = 5
    prog_bar = ProgressBar(n, "Computing: ")
    for p in range(n+10):
        next(prog_bar)
        time.sleep(0.1) # any code you need
    print("")
    print("Done iteration")

def tests():
    _test_prog_bar_as_iterator()
    _test_prog_bar_as_object()
    print("Done tests.")

if __name__ == "__main__":
    tests()