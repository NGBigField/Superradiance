# ==================================================================================== #
# |                                 Imports                                          | #
# ==================================================================================== #

# For plotting:
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib import cm
from matplotlib.cm import ScalarMappable
from qutip.matplotlib_utilities import complex_phase_cmap
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.axes import Axes
from matplotlib.transforms import Bbox

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
    ClassVar,
)

# Import our tools and utils:
try:
    from utils import (
        strings,
        args,
        saveload,
        assertions,
    )
except ImportError:
    import strings
    import args
    import saveload
    import assertions
    

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

def close_all():
    plt.close('all')

def draw_now():
    plt.show(block=False)
    plt.pause(0.1)

def new_axis(is_3d:bool=False):
    fig = plt.figure()
    if is_3d:
        axis : Axes3D = _axes3D(fig, azim=-35, elev=35)
    else:
        axis : Axes = plt.axes(fig)
    return axis

def save_figure(fig:Optional[Figure]=None, file_name:Optional[str]=None ) -> None:
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

def plot_wigner_bloch_sphere(rho:np.matrix, num_points:int=100, ax:Axes=None, colorbar_ax:Axes=None, warn_imaginary_part:bool=False, title:str=None, with_axes_arrows:bool=True) :
    # Constants:
    radius = 1

    # Check inputs:
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
    else:
        fig = ax.figure
        
    # Basic data:
    num_atoms = rho.shape[0] - 1
    phi = np.linspace(0, 2 * np.pi, 2 * num_points)
    theta = np.linspace(0, np.pi, num_points)
    theta, phi = np.meshgrid(theta, phi)
    X = np.sin(theta) * np.cos(phi) * radius
    Y = np.sin(theta) * np.sin(phi) * radius
    Z = np.cos(theta) * radius
    W = np.zeros(np.shape(phi))
    j = num_atoms / 2

    # Basic functions:
    floor = math.floor
    
    # Iterate:
    k_vals = np.linspace(0, 2*j, floor(2*j + 1))
    pb = ProgressBar(len(k_vals), print_prefix="calculating wigner-function")
    for k in k_vals :
        pb.next()
        
        for q in np.linspace(-k, k, floor(2 * k + 1)):

            if q >= 0:
                Ykq = sph_harm(q, k, phi, theta)
            else:
                Ykq = (-1)**q*np.conj(sph_harm(-q, k, phi, theta))
            Gkq = 0
            
            for m1 in np.linspace(-j, j, floor(2 * j + 1)):
                for m2 in np.linspace(-j, j, floor(2 * j + 1)):
                    
                    if -m1 + m2 + q == 0:
                        tracem1m2 = rho[floor(m1 + j), floor(m2 + j)]
                        wig_sym = wigner_3j(j, k, j, -m1, q, m2)
                        wig_val = np.conj(complex(wig_sym))
                        Gkq = Gkq + tracem1m2 * np.sqrt(2 * k + 1) * (-1) ** (j - m1) * wig_val
            W = W + Ykq * Gkq;
    pb.close()

    if warn_imaginary_part and ( np.max(abs(np.imag(W))) > 1e-3 ):
        print('The wigner function has non negligible imaginary part ', str(np.max(abs(np.imag(W)))))
    W = np.real(W)

    fmax, fmin = W.max(), W.min()

    fcolors = W / np.max(np.abs(W))
    # Set the aspect ratio to 1 so our sphere looks spherical
    surface_plot = ax.plot_surface(X, Y, Z, rstride=1, cstride=1, facecolors=cm.bwr(fcolors / 2 + 0.5))
    m = cm.ScalarMappable(cmap=cm.bwr)
    m.set_array(fcolors)
    m.set_clim(-min(np.max(np.abs(W)),2), min(np.max(np.abs(W)),2))
    
    if title is not None:
        ax.set_title(title, y=1.10)

    # Color bar:
    if colorbar_ax is None:
        colorbar = plt.colorbar(mappable=m, shrink=0.5, ax=ax)
    else:
        colorbar = plt.colorbar(m, ax=ax, cax=colorbar_ax, shrink=0.5)        

    # xyz axes:
    if with_axes_arrows:
        for i, str_ in enumerate(['x', 'y', 'z']):
            xyz = [0, 0, 0]
            xyz[i] = 1.5
            quiver_inputs = [0,0,0]
            quiver_inputs.extend(xyz)
            arrow = ax.quiver(*quiver_inputs,  length=1.2, arrow_length_ratio=0.1, zorder=100+i, alpha=0.5)
            arrow.set_linewidth(4)
            xyz[i] = 1.7
            ax.text(*xyz, str_, font=dict(size=18))


    ax.set_title('$W(\\theta,\phi)$', fontsize=16)
    # Turn off the axis planes
    ax.set_axis_off()
    return surface_plot

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
    plt.sca(ax)
    plt.xticks( pos_range, [ f"|{m}>" for m in m_range] )
    plt.yticks( pos_range, [ f"<{m}|" for m in m_range] )

    # Colorbar:
    left, bottom, width, height = 0.95, 0.1, 0.02, 0.8
    cax = plt.axes([left, bottom, width, height])
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

def plot_matter_state(state:np.matrix, block_sphere_resolution:int=100):
    matter_state_obj = MatterStatePlot(initial_state=state, block_sphere_resolution=block_sphere_resolution, show_now=True)
    return matter_state_obj.figure

# ==================================================================================== #
#|                                     Classes                                        |#
# ==================================================================================== #

class MatterStatePlot():

    _separate_colorbar_axis : ClassVar[bool] = True

    def __init__(self, block_sphere_resolution:int=100, initial_state:Optional[np.matrix]=None, show_now:bool=False) -> None:
        self.block_sphere_resolution = block_sphere_resolution
        fig, ax1, ax2, ax3 = MatterStatePlot._init_figure()
        self.axis_bloch_sphere : Axes3D = ax1
        self.axis_bloch_sphere_colorbar : Axes = ax2
        self.axis_block_city : Axes3D = ax3
        self.figure : Figure = fig
        if initial_state is not None:
            self.update(initial_state, title="Initial-State", show_now=show_now)
    
    def update(self, state:np.matrix, title:Optional[str]=None, score_str:Optional[str]=None, fontsize:int=16, show_now:bool=False) -> None:
        assertions.density_matrix(state, robust_check=False)
        self.refresh_figure()
        plot_wigner_bloch_sphere(state, ax=self.axis_bloch_sphere, num_points=self.block_sphere_resolution, colorbar_ax=self.axis_bloch_sphere_colorbar)
        plot_city(state, ax=self.axis_block_city)
        if title is not None:
            self.figure.suptitle(title, fontsize=fontsize)
        if show_now:
            draw_now()
        if score_str is not None:
            self.axis_bloch_sphere.text(x=-0.5 ,y=-0.5, z=-2, s=score_str, fontsize=12)
    
    def close(self) -> None:
        plt.close(self.figure)
            
    def refresh_figure(self) -> None :
        plt.figure(self.figure.number)
        plt.clf()
        fig, ax1, ax2, ax3 = MatterStatePlot._init_figure(self.figure)
        self.axis_bloch_sphere : Axes3D = ax1
        self.axis_bloch_sphere_colorbar : Axes = ax2
        self.axis_block_city : Axes3D = ax3
        self.figure : Figure = fig
        
    @staticmethod
    def _init_figure(fig:Optional[Figure]=None):
        # Control
        separate_colorbar_axis : bool = MatterStatePlot._separate_colorbar_axis
        # fig:
        if fig is None:
            fig = plt.figure(figsize=(11,6))
        # Create axes:
        ax_block_city = _axes3D(fig)
        ax_bloch_sphe = fig.add_subplot(1,2,1, projection='3d')        
        if separate_colorbar_axis:
            ax_color_bar = fig.add_axes([0.0, 0.1, 0.02, 0.8])
        else:
            ax_color_bar = None
        # set dims:
        ax_bloch_sphe.set_position(Bbox([[0.00, 0.0], [0.45, 1.0]])) 
        ax_block_city.set_position(Bbox([[0.45, 0.0], [0.95, 0.9]]))  
        # Return:
        return fig, ax_bloch_sphe, ax_color_bar, ax_block_city

class VideoRecorder():
    def __init__(self, fps:float=10.0) -> None:
        self.fps = fps
        self.frames_dir : str = self._reset_temp_folders_dir()
        self.frames_duration : List[int] = []
        self.frames_counter : int = 0

    def capture(self, fig:Optional[Figure]=None, duration:Optional[int]=None)->None:
        # Complete missing inputs:
        duration = args.default_value(duration, 1)
        if fig is None:
            fig = plt.gcf()
        # Check inputs:
        assertions.integer(duration, reason=f"duration must be an integer - meaning the number of frames to repeat a single shot")
        # Prepare data
        fullpath = self.crnt_frame_path
        # Set the current figure:
        plt.figure(fig.number)
        # Capture:
        plt.savefig(fullpath)
        # Update:
        self.frames_counter += 1
        self.frames_duration.append(duration)

    def write_video(self, name:Optional[str]=None)->None:
        # Complete missing inputs:
        name = args.default_value(name, default_factory=strings.time_stamp )        
        # Prepare folder for video:
        saveload.make_sure_folder_exists(VIDEOS_FOLDER)
        clips_gen = self.image_clips()
        video_slides = concatenate_videoclips( list(clips_gen), method='chain' )
        # Write video file:
        fullpath = VIDEOS_FOLDER+name+".mp4"
        video_slides.write_videofile(fullpath, fps=self.fps)

    @property
    def crnt_frame_path(self) -> str:         
        return self._get_frame_path(self.frames_counter)

    def image_clips(self) -> Generator[ImageClip, None, None] :
        base_duration = 1/self.fps
        for img_path, frame_duration in zip( self.image_paths(), self.frames_duration ):
            yield ImageClip(img_path+".png", duration=base_duration*frame_duration)

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
        self.expected_end :int = expected_end
        self.print_prefix :int = print_prefix
        self.print_length :int = print_length
        self.print_out = print_out
        self.counter = 0
        self._as_iterator : bool = False

    def __next__(self) -> int:
        return self.next()

    def __iter__(self):
        self._as_iterator = True
        return self

    def next(self, increment:int=1, extra_str:Optional[str]=None) -> int:
        self.counter += increment
        if self._as_iterator and self.counter > self.expected_end:
            self.close()
            raise StopIteration
        self._show(extra_str)
        return self.counter

    def close(self):
        full_bar_length = self.print_length+len(self.print_prefix)+4+len(str(self.expected_end))*2
        print(
            f"{(' '*(full_bar_length))}", 
            end='\r', 
            file=self.print_out, 
            flush=True
        )

    def _show(self, extra_str:Optional[str]=None):
        # Unpack properties:
        i = self.counter
        prefix = self.print_prefix
        expected_end = int( self.expected_end )
        print_length = int( self.print_length )

        # Derive print:
        if i>expected_end:
            crnt_bar_length = print_length
        else:
            crnt_bar_length = int(print_length*i/expected_end)
        s = f"{prefix}[{u'█'*crnt_bar_length}{('.'*(print_length-crnt_bar_length))}] {i:d}/{expected_end:d}"

        if extra_str is not None:
            s += " "+extra_str

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

def _test_bloch_sphere():
    # Specific imports:
    import pathlib, sys
    sys.path.append(str(pathlib.Path(__file__).parent.parent))
    from fock import Fock
    # Figure:
    plt.show(block=False)
    fig = plt.figure(figsize=(11,6))
    ax1 = fig.add_subplot(1,2,1, projection='3d')
    # ax2 = fig.add_subplot(1,2,2, projection='3d')
    ax2 = _axes3D(fig)
    ax1.set_position(Bbox([[0.0, 0.2], [0.45, 0.8]])) 
    ax2.set_position(Bbox([[0.4, 0.0], [0.95, 1.0]]))    
    # Define state:
    num_moments = 2
    initial_state = Fock.create_coherent_state(num_moments=num_moments, alpha=0, output='density_matrix')
    # Plot:
    plot_city(initial_state, ax=ax2)
    plot_wigner_bloch_sphere(initial_state, ax=ax1, num_points=10)
    # Done:
    plt.show(block=False)
    print("Finished print")

def _test_bloch_sphere_object():
    # Specific imports:
    import pathlib, sys
    sys.path.append(str(pathlib.Path(__file__).parent.parent))
    from fock import Fock
    from coherentcontrol import CoherentControl
    # Define state:
    num_moments = 2
    initial_state = Fock.create_coherent_state(num_moments=num_moments, alpha=0, output='density_matrix')
    # Init figure object:
    state_fig = MatterStatePlot(block_sphere_resolution=10, show_now=True)
    state_fig.update(initial_state, title="Ground State", show_now=True)
    plt.show(block=False)
    # apply pulse:
    coherent_control = CoherentControl(num_moments=num_moments)
    final_state = coherent_control.pulse_on_state(initial_state, x=0.56)
    state_fig.update(final_state, title="Final State", show_now=True)
    # Show
    print(f"Plotted")
    

def _test_gkp_state():
    # Constants:
    num_moments = 40
    # Imports:
    from pathlib import Path
    sys.path.append( str(Path(__file__).parent.parent) )
    from gkp import goal_gkp_state
    # Get and plot:
    gkp_state = goal_gkp_state(num_moments=num_moments)
    plot_city(gkp_state)
    plot_wigner_bloch_sphere(gkp_state, num_points=200)
    draw_now()
    # Print:
    print("Plotted GKP State")

def tests():
    # _test_prog_bar_as_iterator()
    # _test_prog_bar_as_object()
    # _test_bloch_sphere()
    # _test_bloch_sphere_object()
    _test_gkp_state()
    
    print("Done tests.")

if __name__ == "__main__":
    tests()
    print("Done.")