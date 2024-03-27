# ==================================================================================== #
#| Imports:                                         
# ==================================================================================== #

if __name__ == "__main__":
    import sys, pathlib
    sys.path.append(
        pathlib.Path(__file__).parent.parent.parent.__str__()
    )
# For type annotations:
from typing import Callable

# Import useful types:
from algo.optimization import BaseParamType
from algo.coherentcontrol import Operation

# Basic algo mechanisms:
from algo.coherentcontrol import CoherentControl
from physics.famous_density_matrices import ground_state

# for numerics:
import numpy as np
from numpy import pi

# for plotting:
from utils.visuals import plot_matter_state, plot_wigner_bloch_sphere, plot_plain_wigner, ViewingAngles, BlochSphereConfig, save_figure, draw_now
from utils import assertions, saveload
import matplotlib.pyplot as plt

# for printing progress:
from utils import strings

# for enums:
from enum import Enum, auto

# for sleeping:
from time import sleep

# For emitted light calculation 
from physics.emitted_light_approx import main as calc_emitted_light

# For writing results to file:
from csv import DictWriter

# ==================================================================================== #
#| Constants:
# ==================================================================================== #

# DEFAULT_COLORLIM = (-0.1, 0.2)
DEFAULT_COLORLIM = None


def _get_movie_config(
    create_movie:bool, num_transition_frames:int
) -> CoherentControl.MovieConfig:
    # Basic data:
    fps=30
    
    bloch_sphere_config = BlochSphereConfig(
        alpha_min=0.1,
        resolution=150,
        viewing_angles=ViewingAngles(
            elev=+10,
            azim=+45
        )
    )
    
    # Movie config:
    movie_config=CoherentControl.MovieConfig(
        active=create_movie,
        show_now=False,
        num_freeze_frames=fps//2,
        fps=fps,
        bloch_sphere_config=bloch_sphere_config,
        num_transition_frames=num_transition_frames,
        temp_dir_name="temp_movie"     
    )
    
    return movie_config


def create_movie(
    num_atoms:int = 40,
    num_transition_frames = 50
):
    # Start:
    print(f"Creating Movie...")

    # General variables:
    coherent_control = CoherentControl(num_atoms)
    movie_config = _get_movie_config(True, num_transition_frames)

    ## Sequence:
    initial_state = ground_state(num_atoms)
    # Operations:
    y1 = coherent_control.standard_operations(num_transition_frames).power_pulse_on_specific_directions(1, [1])
    z2 = coherent_control.standard_operations(num_transition_frames).power_pulse_on_specific_directions(2, [2])
    x2 = coherent_control.standard_operations(num_transition_frames).power_pulse_on_specific_directions(2, [0])
    operations : list[Operation] = []
    operations.append( x2 )

    standard_operations : CoherentControl.StandardOperations  = coherent_control.standard_operations(num_intermediate_states=num_transition_frames)
    rotation  = standard_operations.power_pulse_on_specific_directions(power=1, indices=[0, 1, 2])
    squeezing = standard_operations.power_pulse_on_specific_directions(power=2, indices=[0, 1])

    theta = [
        +0.8937567499106599 , +3.2085033698137830 , -2.3242661423839071 , +0.0036751816770657 , -0.7836001773757240 , 
        +2.6065083924231915 , +2.2505047554207338 , -2.4740789195081394        
    ] # 1 step - 96 fidelity

    operations  = [
        rotation, squeezing, rotation
    ]


    # Theta:
    # theta = [pi/2]
    
    # create state:
    final_state = coherent_control.custom_sequence(state=initial_state, theta=theta, operations=operations, movie_config=movie_config)
    # plot_matter_state(final_state)
    
    # Finish
    print("Done with movie.")



if __name__ == "__main__":
    create_movie()
    
    print("Done.")