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
        case "low_squeezing":
            theta = [
                +0.4935717071225814 , +1.7488554910458203 , +2.4081075251579991 , +0.1960856683031602 , +0.0868691825942013 ,
                +0.7972821358300305 , +0.1037948476124894 , +2.9860069022070066 , -0.1259184534263386 , +0.0416548691243354 ,
                +0.8792154967902378 , +1.0851850619300669 , +0.3040529641594746 , +0.0574856817405069 , +0.0210535691786186 ,
                +0.7827130861085472 , -0.3885524067827190 , -0.1380271437296924 , +0.0767853562766987 , -0.1174564141277240 ,
                +0.4576062201652379 , +1.2726065155223425 , +1.1641048116223858 , +0.0212355775098413 , +0.0844961998343022 ,
                +0.2787371711063600 , +0.9041926789581701 , +1.7690041530910321 , -0.0250755863494547 , +0.0076069776716833 ,
                +0.6157074876395792 , +0.6264536407238972 , +0.3242211271688092
            ]
            operations  = [
                rotation, p2_pulse
            ] * 6 + [rotation]

        case "high_squeezing":
            theta = [
                -0.0270826178762894 , +0.0420785292642372 , +1.6552895039760658 , +1.5101402120324945 , -0.0008950464974893 ,
                -0.1450707540879570 , +0.2292109730479739 , +1.5551756612995788 , -0.0233353251592546 , +0.0233642260889000 ,
                +0.0014409019759466 , +1.1375470015048958 , +0.0108297389140748 , +0.0000000784248237 , +0.0000001242354474 ,
                +0.1083037282137895 , +0.1067735081262144 , -0.6376625013026994 , +0.0000002425902640 , +0.0000001562681915 ,
                +0.0460796592248603 , +0.0485811088429098 , +0.0169410436835062 , +0.0000005761451667 , +0.0000004278663367 ,
                +0.0251369727619576 , +0.1486675404429519 , +0.1559500007126816 , +0.0000001227203463 , +0.0000000310174803 ,
                +0.0066005859930512 , +0.0805438764732726 , +0.2188462320291897
            ] # fidelity 0.84 - 6 steps
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
    resolution:int,
    show_now:bool
) -> CoherentControl.MovieConfig:
    # Basic data:
    fps=30
    
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
        temp_dir_name="temp_movie"+strings.time_stamp()    
    )
    
    return movie_config


def _get_target_state(num_atoms:int):
    return cat_state(num_atoms, num_legs=2, phase=np.pi/2)



def create_movie(
    which: _WhichVersionLiteral,
    plot_target:bool = False,
    active_movie:bool = False,
    num_transition_frames:int|tuple[int, int, int] = (60, 180, 240),
    resolution:int = 200,
    show_now:bool = False,
):


    ## Basic inputs and constants:
    num_transition_frames = num_transition_frames if create_movie else 0
    num_atoms:int = 12

    ## get data:
    coherent_control, initial_state, theta, operations, cost_function = _get_type_inputs(which=which, num_atoms=num_atoms, num_intermediate_states=0)
    movie_config = _get_movie_config(active_movie, num_transition_frames, resolution, show_now)    



    def _create_matter_figure(state)->MatterStatePlot:
        return MatterStatePlot(initial_state=state, bloch_sphere_config=movie_config.bloch_sphere_config, horizontal=True)   
    
    if plot_target:
        target = _get_target_state(num_atoms)
        target_plot = _create_matter_figure(target)
        target_plot.set_title("Target state")

    ## Go:
    print(f"Creating Movie...")
    final_state = coherent_control.custom_sequence(state=initial_state, theta=theta, operations=operations, movie_config=movie_config)

    _create_matter_figure(final_state)
    
    # Finish
    print("Done with movie.")


if __name__ == "__main__":
    create_movie(which="low_squeezing")  # "low_squeezing",  "high_squeezing" or "standard"
    
    print("Done.")