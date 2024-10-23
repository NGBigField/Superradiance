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
    create_movie:bool, num_transition_frames:int, horizontal_movie:bool
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
        horizontal_figure=horizontal_movie,
        num_freeze_frames=fps//2,
        fps=fps,
        bloch_sphere_config=bloch_sphere_config,
        num_transition_frames=num_transition_frames,
        temp_dir_name="temp_movie"     
    )
    
    return movie_config


def main(
    what_movie = "squeeze_to_kitten",
    resolution:int = 200
):
    # Start:
    print(f"Creating Movie for movie_settings {what_movie!r}")

    horizontal_movie = True
    
    match what_movie:
        case "cat4":
            num_atoms = 40
            num_transition_frames = 150
        case "cat4_i24":
            num_atoms = 24
            num_transition_frames = 150
        case "rot":
            num_atoms = 10
            num_transition_frames = 70
            horizontal_movie = False
        case "rot_squeeze_rot":
            num_atoms = 10
            num_transition_frames = 100
            horizontal_movie = False
        case "squeeze_x2":
            num_atoms = 10
            num_transition_frames = 100
            horizontal_movie = False
        case "squeeze_y2":
            num_atoms = 10
            num_transition_frames = 100
            horizontal_movie = False
        case "small_rotation":
            num_atoms = 10
            num_transition_frames = 100
            horizontal_movie = False
        case "test":
            num_atoms = 10
            num_transition_frames = 10
        case "squeeze_to_kitten":
            num_atoms = 21
            num_transition_frames = 100
        case _:
            raise ValueError("Choose a state that has an implementation")
    assert isinstance(num_atoms, int), "Choose a state that has an implementation"

    # General variables:
    coherent_control = CoherentControl(num_atoms)
    _pulse_of_power_and_directions = coherent_control.standard_operations(num_transition_frames).power_pulse_on_specific_directions
    movie_config = _get_movie_config(True, num_transition_frames, horizontal_movie)
    movie_config.bloch_sphere_config.resolution = resolution

    if what_movie in ["cat4", "cat_i24"]:
        movie_config.bloch_sphere_config.viewing_angles.azim = -45

    ## Sequence:
    initial_state = ground_state(num_atoms)
    # Operations:
    y1 = _pulse_of_power_and_directions(1, [1])
    xy = _pulse_of_power_and_directions(1, [0, 1])
    x2 = _pulse_of_power_and_directions(2, [0])
    y2 = _pulse_of_power_and_directions(2, [1])
    z2 = _pulse_of_power_and_directions(2, [2])
    operations : list[Operation] = []

    standard_operations : CoherentControl.StandardOperations  = coherent_control.standard_operations(num_intermediate_states=num_transition_frames)
    rotation  = standard_operations.power_pulse_on_specific_directions(power=1, indices=[0, 1, 2])
    squeezing = standard_operations.power_pulse_on_specific_directions(power=2, indices=[0, 1])

    match what_movie:
        case "cat4":
            operations  = [rotation, squeezing, rotation]
            theta = [
                +0.8937567499106599 , +3.2085033698137830 , -2.3242661423839071 , +0.0036751816770657 , -0.7836001773757240 , 
                +2.6065083924231915 , +2.2505047554207338 , -2.4740789195081394        
            ] 
        case "cat4_i24":
            operations  = [rotation, squeezing, rotation]
            theta = [
                +1.1634502923694394 , +0.7391690305215712 , -3.2224714983636460 , -0.0041129266235580 , -0.7896391684875264 , 
                +0.9809815207185668 , +2.4048152268109622 , -2.6451848191261678     
            ] 
        case "rot":
            operations = [ y1  ]*4 
            theta      = [-pi/2]*4 

        case "rot_squeeze_rot":
            operations = [ y1  , z2  , z2  ,  y1   ] 
            theta      = [-pi/2, pi/2, pi/2, -pi/2 ] 

        case "squeeze_x2":
            operations = [x2  ]*4 
            theta      = [pi/2]*4 

        case "squeeze_y2":
            # Another intial state
            initial_state = coherent_control.custom_sequence(state=initial_state, theta=[-pi/2], operations=[y1], movie_config=None) 
            # Params:
            operations = [y2 ]*3
            theta      = [+0.25, -0.5, +0.25]

        case "small_rotation":
            # Another intial state
            initial_state = coherent_control.custom_sequence(state=initial_state, theta=[-pi/2], operations=[y1], movie_config=None) 
            # Params:
            operations = [xy ]
            theta      = [+1.4*pi, -1.4*pi]

        case "squeeze_to_kitten":
            operations = [ y1        , z2  ] 
            theta      = [-pi/2 - 0.6, pi/2] 

        case "test":
            operations = [y1  ] 
            theta      = [-pi ] 


    # create state:
    final_state = coherent_control.custom_sequence(state=initial_state, theta=theta, operations=operations, movie_config=movie_config)
    # plot_matter_state(final_state)
    
    # Finish
    print("Done with movie.")



if __name__ == "__main__":
    main()
    
    print("Done.")