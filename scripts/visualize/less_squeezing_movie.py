# ==================================================================================== #
#| Imports:                                         
# ==================================================================================== #

if __name__ == "__main__":
    import sys, pathlib
    sys.path.append(
        pathlib.Path(__file__).parent.parent.parent.__str__()
    )

# For type annotations:
from typing import Callable, Literal, TypeAlias

# Import useful types:
from algo.optimization import BaseParamType
from algo.optimization import (
    FixedParam, 
    FreeParam,
    BaseParamType,
    Operation
)

# Basic algo mechanisms:
from algo.coherentcontrol import CoherentControl

# Physical states:
from physics.famous_density_matrices import ground_state, cat_state
from algo.common_cost_functions import fidelity_to_cat

# for numerics:
import numpy as np
from numpy import pi

# for plotting:
from utils.visuals import ViewingAngles, BlochSphereConfig, save_figure, \
    draw_now, MatterStatePlot, VideoRecorder, VIDEOS_FOLDER, ImageClip, concatenate_videoclips

from utils import assertions, saveload
from algo.coherentcontrol import SequenceMovieRecorder
import matplotlib.pyplot as plt

# for printing progress:
from utils import strings, files

# For listing files in folder and navigating data:
import os


_WhichVersionLiteral : TypeAlias = Literal["low_squeezing", "high_squeezing", "standard"]


# ==================================================================================== #
#| Constants:
# ==================================================================================== #

# DEFAULT_COLORLIM = (-0.1, 0.2)
DEFAULT_COLORLIM = None

def _get_params(
    which, 
    num_atoms, 
    num_intermediate_states
) -> tuple[
    list[BaseParamType],
    list[Operation],
    CoherentControl,
    Callable[[np.matrix], float]
]:
    # Get the coherent control object:
    coherent_control = CoherentControl(num_atoms=num_atoms)
    
    # Get the standard operations:
    standard_operations : CoherentControl.StandardOperations = coherent_control.standard_operations(num_intermediate_states=num_intermediate_states)
    rotation    = standard_operations.power_pulse_on_specific_directions(power=1, indices=[0, 1, 2])
    p2_pulse    = standard_operations.power_pulse_on_specific_directions(power=2, indices=[0, 1])    

    # Get the operations:
    match which:
        case "high_squeezing":
            theta = [
                +2.9501274976023235 , +3.2399114064359571 , +0.6711011017455951 , -0.9800698587987760 , +1.9738650123692736 , 
                +2.3127010356771458 , +0.7945229709481191 , +3.2415341143808769 , -1.2590767381798811 , -0.4016639825148159 , 
                +1.2599969488800804 , -0.2290788211072962 , +1.1336332129353892 , -0.1862593986082912 , +3.2197074096647107 , 
                +1.0446454901441966 , +1.0081675551217884 , -0.6398380319328609 , +0.0339691513333506 , +1.9846343164347613 , 
                +0.3862605688445813 , +0.6948918553266052 , +2.6149161854848124 , -0.0816392408593655 , +2.6528924481384344 , 
                +1.0352190047584462 , +0.1434425827366452 , +3.1679887515691787 , +0.6467276638480792 , +1.2442019988716795 , 
                -0.4302027448542042 , +2.6942912321377754 , +0.4213405637758895 
            ] # fidelity 0.9999000
            operations  = [
                rotation, p2_pulse
            ] * 6 + [rotation]

        case "low_squeezing":
            # fidelity = 0.992312306468  squeezing = 0.646951683761
            theta = [
                +2.8567573275402358 , +2.3530728677226458 , +2.2439229815723869 , +0.1821548064098444 , +0.0716951667788887 ,
                +1.1530588927182843 , +0.3038442874073793 , +2.4789842872232315 , -0.0710739128659769 , +0.0560833338984745 ,
                +0.6579188063783105 , +1.0725239709669492 , -0.1056646686414356 , +0.1015339259605154 , +0.0261682335557583 ,
                +1.0564150184662844 , -0.5078251013379605 , -0.1469865359536207 , +0.0832787357668058 , -0.1077822220912470 ,
                +0.5860337832017745 , +1.1728652677937685 , +0.9667368248421768 , +0.0396459094821724 , +0.0880697870480867 ,
                -0.0244978653355885 , +0.6448744819710170 , +1.3478740653368053 , +0.0136571295627677 , +0.0185285994375842 ,
                +1.0660199114245268 , +1.0563404413843753 , +0.7897371345515618
            ]
            operations  = [
                rotation, p2_pulse
            ] * 6 + [rotation]

        case "standard":
            theta = [
                +0.0, +0.0 , +0.0 , +pi/2 , +0 ,   #1 
                +0.0, +0.0, +pi/2, 0.0, 0.0,   #2
                +0.0, +pi/2, +0.0
            ] # fidelity 0.9918 - 6 steps
            operations  = [
                rotation, p2_pulse
            ] * 2 + [rotation]
    
    param_config : list[BaseParamType] = []
    for i, value in enumerate(theta):        
        this_config = FixedParam(index=i, value=value)
        param_config.append(this_config)


    # Get the cost function:
    cost_function = fidelity_to_cat(num_atoms, num_legs=2, phase=np.pi/2)

    # Return the parameters:
    return param_config, operations, coherent_control, cost_function


def _get_type_inputs(
    which:_WhichVersionLiteral, num_atoms:int, num_intermediate_states:int
) -> tuple[
    CoherentControl,
    np.matrix,
    list[float],
    list[Operation],
    Callable[[np.matrix], float]
]:
    # Get all needed data:
    params, operations, coherent_control, cost_function = _get_params(which, num_atoms, num_intermediate_states)
    initial_state = ground_state(num_atoms=num_atoms)
    
    # derive theta:
    theta = [param.get_value() for param in params]
    
    return coherent_control, initial_state, theta, operations, cost_function



def _get_movie_config(
    active:bool,
    num_transition_frames:int|tuple[int, int, int],
    fps:int,
    resolution:int,
    show_now:bool,
    name:str,
) -> CoherentControl.MovieConfig:
    # Basic data:
    bloch_sphere_config = BlochSphereConfig(
        alpha_min=0.1,
        resolution=resolution,
        viewing_angles=ViewingAngles(
            elev=+10,
            azim=+45
        )
    )
    
    # Movie config:
    movie_config=CoherentControl.MovieConfig(
        active=active,
        show_now=show_now,
        num_freeze_frames=fps//2,
        fps=fps,
        bloch_sphere_config=bloch_sphere_config,
        num_transition_frames=num_transition_frames,
        temp_dir_name="temp_movie"+strings.time_stamp(),
        video_name=name
    )
    
    return movie_config


def _get_target_state(num_atoms:int):
    return cat_state(num_atoms, num_legs=2, phase=np.pi/2)



def create_movie(
    which: _WhichVersionLiteral,
    active_movie:bool = True,
    plot_target_and_final:bool = False,
    num_transition_frames:int|tuple[int, int, int] = (60, 100, 200),
    resolution:int = 150,
    show_now:bool = False,
):


    ## Basic inputs and constants:
    num_transition_frames = num_transition_frames if active_movie else 0
    num_atoms:int = 12
    fps:int = 30

    ## get data:
    coherent_control, initial_state, theta, operations, cost_function = _get_type_inputs(which=which, num_atoms=num_atoms, num_intermediate_states=0)
    movie_config = _get_movie_config(active_movie, num_transition_frames, fps, resolution, show_now, which)    



    def _create_matter_figure(state)->MatterStatePlot:
        return MatterStatePlot(initial_state=state, bloch_sphere_config=movie_config.bloch_sphere_config, horizontal=True)   
    
    if plot_target_and_final:
        target = _get_target_state(num_atoms)
        target_plot = _create_matter_figure(target)
        target_plot.set_title("Target state")

        final_state = coherent_control.custom_sequence(state=initial_state, theta=theta, operations=operations, movie_config=None)
        final_plot = _create_matter_figure(final_state)
        final_plot.set_title("Final state")

        draw_now()

    ## Go:
    print(f"Creating Movie...")
    final_state = coherent_control.custom_sequence(state=initial_state, theta=theta, operations=operations, movie_config=movie_config)
    
    # Finish
    print("Done with movie.")


if __name__ == "__main__":
    create_movie(which="low_squeezing")  # "low_squeezing",  "high_squeezing" or "standard"
    
    print("Done.")