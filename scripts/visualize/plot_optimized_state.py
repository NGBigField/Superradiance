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

# Our best optimized results:    
from scripts.optimize.cat4       import best_sequence_params as cat4_params
from scripts.optimize.cat2       import best_sequence_params as cat2_params
from scripts.optimize.gkp_hex    import best_sequence_params as gkp_hex_params
from scripts.optimize.gkp_square import best_sequence_params as gkp_square_params

# Cost function:
from algo.common_cost_functions import fidelity_to_cat, fidelity_to_gkp

# for plotting:
from utils.visuals import plot_matter_state, plot_wigner_bloch_sphere, plot_light_wigner, ViewingAngles, BlochSphereConfig, save_figure
from utils import assertions
import matplotlib.pyplot as plt

# for printing progress:
from utils import strings

# for enums:
from enum import Enum, auto

# for sleeping:
from time import sleep


# ==================================================================================== #
#| Helper types:
# ==================================================================================== #

class StateType(Enum):
    GKPHex = auto()
    GKPSquare = auto()
    Cat2 = auto()
    Cat4 = auto()


# ==================================================================================== #
#| Inner Functions:
# ==================================================================================== #

def _get_best_params(
    type_:StateType, 
    num_atoms:int,
    num_intermediate_states:int
) -> tuple[
    list[BaseParamType],
    list[Operation]
]:
    if type_ is StateType.GKPHex:
        return gkp_hex_params(num_atoms, num_intermediate_states=num_intermediate_states)
    elif type_ is StateType.GKPSquare:
        return gkp_square_params(num_atoms, num_intermediate_states=num_intermediate_states)
    elif type_ is StateType.Cat4:
        return cat4_params(num_atoms, num_intermediate_states=num_intermediate_states)
    elif type_ is StateType.Cat2:
        return cat2_params(num_atoms, num_intermediate_states=num_intermediate_states)
    else:
        raise ValueError(f"Not an option '{type_}'")


def _get_type_inputs(
    type_:StateType, num_atoms:int, num_intermediate_states:int
) -> tuple[
    CoherentControl,
    np.matrix,
    list[float],
    list[Operation],
    Callable[[np.matrix], float]
]:
    # Get all needed data:
    params, operations = _get_best_params(type_, num_atoms, num_intermediate_states)
    initial_state = ground_state(num_atoms=num_atoms)
    coherent_control = CoherentControl(num_atoms=num_atoms)
    cost_function = _get_cost_function(type_=type_, num_atoms=num_atoms)
    
    # derive theta:
    theta = [param.get_value() for param in params]
    
    return coherent_control, initial_state, theta, operations, cost_function
   
   
def _get_cost_function(type_:StateType, num_atoms:int) -> Callable[[np.matrix], float]:
    if type_ is StateType.GKPHex:
        return fidelity_to_gkp(num_atoms=num_atoms, gkp_form="hex")
    elif type_ is StateType.GKPSquare:
        return fidelity_to_gkp(num_atoms=num_atoms, gkp_form="square")        
    elif type_ is StateType.Cat4:
        return fidelity_to_cat(num_atoms=num_atoms, num_legs=4)
    elif type_ is StateType.Cat2:
        return fidelity_to_cat(num_atoms=num_atoms, num_legs=2)        
    else:
        raise ValueError(f"Not an option '{type_}'")

def _print_fidelity(final_state:np.matrix, cost_function:Callable[[np.matrix], float]):
    cost = cost_function(final_state)
    fidelity = -cost
    print(f"Fidelity = {fidelity}")
    
        
def _get_movie_config(
    create_movie:bool, num_transition_frames:int, state_type:StateType
) -> CoherentControl.MovieConfig:
    # Basic data:
    fps=20
    
    bloch_sphere_config = BlochSphereConfig(
        alpha_min=0.2,
        resolution=250,
        viewing_angles=ViewingAngles(
            elev=-45
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
        temp_dir_name=state_type.name     
    )
    
    return movie_config

# ==================================================================================== #
#| Declared Functions:
# ==================================================================================== #



def plot_sequence(
    type_:StateType = StateType.GKPSquare,
    num_atoms:int = 40,
    folder:str|None = None
):
    # constants:

    # get
    coherent_control, initial_state, theta, operations, cost_function = _get_type_inputs(type_=type_, num_atoms=num_atoms, num_intermediate_states=0)
    
    # derive:
    n = assertions.integer( (len(operations)-1)/2 )
    
    # iterate: 
    for i in range(n+1):
        print(strings.num_out_of_num(i, n))

        # derive params for this iteration:
        if i==0:
            theta_i = []
            operations_i = []
        else:
            theta_i = theta[:i*5+3]
            operations_i = operations[:i*2+1]
        name = type_.name+f"-{i:02}"

        # Get state:
        state_i = coherent_control.custom_sequence(state=initial_state, theta=theta_i, operations=operations_i)
    
        # plot light:
        plot_light_wigner(state_i)
        save_figure(folder=folder, file_name=name+" - Light")        

        # plot bloch:
        plot_wigner_bloch_sphere(state_i, view_elev=-90, alpha_min=1, title="", num_points=200)
        save_figure(folder=folder, file_name=name+" - Sphere")
        
        # Sleep and close open figures:
        sleep(1)
        plt.close("all")

## Main:
def plot_all_best_results(
    create_movie:bool = False,
    num_atoms:int = 40
):
    # derive:
    num_transition_frames = 20 if create_movie else 0
    
    for type_ in StateType:
        # Print:
        print(f"State: {type_.name!r}")

        # get
        coherent_control, initial_state, theta, operations, cost_function = _get_type_inputs(type_=type_, num_atoms=num_atoms, num_intermediate_states=num_transition_frames)
        movie_config = _get_movie_config(create_movie, num_transition_frames)
        
        # create state:
        final_state = coherent_control.custom_sequence(state=initial_state, theta=theta, operations=operations, movie_config=movie_config)
        
        # print  fidelity:
        _print_fidelity(final_state, cost_function)
        
        # plot light:
        plot_light_wigner(final_state)
        save_figure(file_name=type_.name+" - Light")
        
        # plot bloch:
        # plot_wigner_bloch_sphere(final_state, view_elev=-90, alpha_min=1, title="", num_points=300)
        # save_figure(file_name=type_.name+" - Sphere")
        
        # # plot complete matter:
        # bloch_config = BlochSphereConfig()
        # plot_matter_state(final_state, config=bloch_config)
        # save_figure(file_name=type_.name+" - Matter")
        
        # Done:
        print("Done.")
    

## Main:
def create_movie(
    state_type:StateType = StateType.GKPHex,
    num_atoms:int = 40,
    num_transition_frames = 40
):
    # derive:
    
    # Print:
    print(f"State: {state_type.name!r}")

    # get
    coherent_control, initial_state, theta, operations, cost_function = _get_type_inputs(type_=state_type, num_atoms=num_atoms, num_intermediate_states=num_transition_frames)
    movie_config = _get_movie_config(True, num_transition_frames, state_type)
    
    # create state:
    final_state = coherent_control.custom_sequence(state=initial_state, theta=theta, operations=operations, movie_config=movie_config)
    
    # print  fidelity:
    _print_fidelity(final_state, cost_function)    
    return cost_function(final_state)





if __name__ == "__main__":
    # plot_sequence()
    # create_movie()
    plot_all_best_results()