# ==================================================================================== #
# |                                 Imports                                          | #
# ==================================================================================== #

# For plotting:
from symbol import argument
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.figure import Figure as FigureType
from matplotlib import cm
from matplotlib.cm import ScalarMappable
from qutip.matplotlib_utilities import complex_phase_cmap
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.axes import Axes

# For defining print std_out or other:
import sys

# Everyone needs numpy in their life and other math stuff:
import numpy as np
import math

# For type hints:
from typing import (
    Any,
    Optional,
    Union,
    Generator,
    List,
)

# Import our tools and utils:
from utils import (
    strings,
    args,
    saveload,
    assertions,
)

# For function version detection:
from packaging.version import parse as parse_version

# Operating System and files:
from pathlib import Path
import os

# For videos:
from moviepy.editor import ImageClip, concatenate_videoclips

# For wigner function on bloch sphere:
from sympy.physics.wigner import wigner_3j
from scipy.special import sph_harm



# ==================================================================================== #
#|                                 Constants                                          |#
# ==================================================================================== #
VIDEOS_FOLDER = os.getcwd()+os.sep+"videos"+os.sep

# ==================================================================================== #
#|                              Inner Functions                                       |#
# ==================================================================================== #

if parse_version(mpl.__version__) >= parse_version('3.4'):
    def _axes3D(fig, *args, **kwargs):
        ax = Axes3D(fig, *args, auto_add_to_figure=False, **kwargs)
        return fig.add_axes(ax)
else:
    def _axes3D(*args, **kwargs):
        return Axes3D(*args, **kwargs)



# ==================================================================================== #
#|                             Declared Functions                                     |#
# ==================================================================================== #

def new_axis(is_3d:bool=False):
    fig = plt.figure()
    if is_3d:
        axis : Axes3D = _axes3D(fig, azim=-35, elev=35)
    else:
        axis : Axes = plt.axes(fig)
    return axis

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

def plot_wigner_bloch_sphere(rho:np.matrix, num_points:int=100, ax:Axes=None) -> None:
    # Basic data:
    num_atoms = rho.shape[0]
    phi = np.linspace(0, 2 * np.pi, 2 * num_points)
    theta = np.linspace(0, np.pi, num_points)
    theta, phi = np.meshgrid(theta, phi)
    X = np.sin(theta) * np.cos(phi)
    Y = np.sin(theta) * np.sin(phi)
    Z = np.cos(theta)
    W = np.zeros(np.shape(phi))
    j = num_atoms / 2

    # Basic functions:
    floor = math.floor
    
    # Iterate:
    for k in np.linspace(0, 2 * j, floor(2 * j + 1)):
        for q in np.linspace(-k, k, floor(2 * k + 1)):

            if q >= 0:
                Ykq = sph_harm(q, k, phi, theta)
            else:
                Ykq = sph_harm(-q, k, phi, theta)
            Gkq = 0
            for m1 in np.linspace(-j, j, floor(2 * j + 1)):
                for m2 in np.linspace(-j, j, floor(2 * j + 1)):
                    if -m1 + m2 + q == 0:
                        tracem1m2 = rho[floor(m1 + j), floor(m2 + j)]
                        Gkq = Gkq + tracem1m2 * np.sqrt(2 * k + 1) * (-1) ** (j - m1) * np.conj(
                            np.complex(wigner_3j(j, k, j, -m1, q, m2)))
            W = W + Ykq * Gkq;

    if np.max(abs(np.imag(W))) > 1e-3:
        print('The wigner function has non negligible imaginary part ', str(np.max(abs(np.imag(W)))))
    W = np.real(W)

    fmax, fmin = W.max(), W.min()

    fcolors = W / np.max(np.abs(W))
    # Set the aspect ratio to 1 so our sphere looks spherical
    fig = plt.figure(figsize=(10,6))
    ax = fig.add_subplot(121, projection='3d')
    a = ax.plot_surface(X, Y, Z, rstride=1, cstride=1, facecolors=cm.bwr(fcolors / 2 + 0.5))
    m = cm.ScalarMappable(cmap=cm.bwr)
    m.set_array(fcolors)
    h = plt.colorbar(m,shrink=0.5)

    h.set_label('$W(\\theta,\phi)$', fontsize=16)
    m.set_clim(-min(np.max(np.abs(W)),2), min(np.max(np.abs(W)),2))
    # Turn off the axis planes
    ax.set_axis_off()
    return fig

def plot_city(mat:Union[np.matrix, np.array], title:Optional[str]=None, ax:Axes=None):
    # Check input type:
    if isinstance(mat, np.matrix):
        mat = np.array(mat)
    assert len(mat.shape)==2
    assert mat.shape[0]==mat.shape[1]

    # Define common symbols:
    pi = np.pi

    n = np.size(mat)
    xpos, ypos = np.meshgrid(range(mat.shape[0]), range(mat.shape[1]))
    xpos = xpos.T.flatten() - 0.5
    ypos = ypos.T.flatten() - 0.5
    zpos = np.zeros(n)
    dx = dy = 0.8 * np.ones(n)
    Mvec = mat.flatten()
    dz = abs(Mvec).tolist()

    # make small numbers real, to avoid random colors
    idx = np.where(abs(Mvec) < 0.001)
    Mvec[idx] = abs(Mvec[idx])

    # Define colors:
    phase_min = -pi
    phase_max = pi
    norm = mpl.colors.Normalize(phase_min, phase_max)
    cmap = complex_phase_cmap()
    colors = cmap(norm(np.angle(Mvec)))

    if ax is None:
        fig = plt.figure()
        ax = _axes3D(fig, azim=-35, elev=35)
    else:
        fig = ax.figure

    ax.bar3d(xpos, ypos, zpos, dx, dy, dz, color=colors)

    if title is not None:
        ax.set_title(title, y=0.95)

    # x axis
    xtics = -0.5 + np.arange(mat.shape[0])
    ax.axes.w_xaxis.set_major_locator(plt.FixedLocator(xtics))
    ax.tick_params(axis='x', labelsize=12)

    # y axis
    ytics = -0.5 + np.arange(mat.shape[1])
    ax.axes.w_yaxis.set_major_locator(plt.FixedLocator(ytics))
    ax.tick_params(axis='y', labelsize=12)

    # z axis
    ax.set_zlim3d([0, 1])  # use min/max

    # Labels:
    M = mat.shape[0]//2
    m_range = range(-M,M+1)
    pos_range = range(mat.shape[0]) 
    plt.xticks( pos_range, [ f"|{m}>" for m in m_range] )
    plt.yticks( pos_range, [ f"<{m}|" for m in m_range] )

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


# ==================================================================================== #
#|                                     Classes                                        |#
# ==================================================================================== #

class VideoRecorder():
    def __init__(self, fps:float=10.0, is_3d:bool=False) -> None:
        self.fps = fps
        self.axis : Union[Axes, Axes3D] = new_axis(is_3d)
        self.frames_dir : str = self._reset_temp_folders_dir()
        self.frames_duration : List[int] = []
        self.frames_counter : int = 0

    def capture(self, ax:Optional[Axes]=None, duration:Optional[int]=None)->None:
        # Complete missing inputs:
        duration = args.default_value(duration, 1)
        # Check inputs:
        assertions.integer(duration, reason=f"duration must be an integer")
        # Prepare data
        ax = args.default_value(ax, self.axis)
        fullpath = self.crnt_frame_path
        # set current axis:
        plt.sca(ax)
        # Capture:
        plt.savefig(fullpath)
        # Update:
        self.frames_counter += 1
        self.frames_duration.append(duration)

    def write_video(self, name:Optional[str]=None)->None:
        # Complete missing inputs:
        name = args.default_value(name, strings.time_stamp() )        
        # Derive basic params:
        base_duration = 1/self.fps
        # Compose video-slides
        video_slides = concatenate_videoclips(
            [ ImageClip(img_path+".png", duration=base_duration*frame_duration) 
             for img_path, frame_duration in zip(self.image_paths(), self.frames_duration) ] , 
            method='compose'
        )
        # exporting final video
        saveload.make_sure_folder_exists(VIDEOS_FOLDER)
        fullpath = VIDEOS_FOLDER+name+".mp4"
        video_slides.write_videofile(fullpath, fps=self.fps)

    @property
    def crnt_frame_path(self) -> str:         
        return self._get_frame_path(self.frames_counter)

    def image_paths(self) -> Generator[str, None, None] :
        for i in range(self.frames_counter):
            yield self._get_frame_path(i)

    def _get_frame_path(self, index:int) -> str:
        return self.frames_dir+"frame"+f"{index}"

    @staticmethod
    def _reset_temp_folders_dir()->str:
        frames_dir = VIDEOS_FOLDER+"temp_frames"+os.sep
        saveload.make_sure_folder_exists(frames_dir)
        return frames_dir



class ProgressBar():

    def __init__(self, expected_end:int, print_prefix:str="", print_length:int=60, print_out=sys.stdout): 
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