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

# Physical states:
from physics.famous_density_matrices import ground_state, cat_state
from physics.gkp import gkp_state

# for numerics:
import numpy as np

# for plotting:
from utils.visuals import ViewingAngles, BlochSphereConfig, save_figure, draw_now, MatterStatePlot, VideoRecorder
from utils import assertions, saveload
from algo.coherentcontrol import SequenceMovieRecorder
import matplotlib.pyplot as plt

# for printing progress:
from utils import strings

# For listing files in folder and navigating data:
import os

from scripts.visualize.plot_optimized_state import StateType, _get_type_inputs

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


def _get_target_state(num_atoms:int, state_type:StateType):
    match state_type:
        case StateType.GKPHex:
            return gkp_state(num_atoms, "square")
        case StateType.Cat2:
            return cat_state(num_atoms, num_legs=2)
        case StateType.Cat4:
            return cat_state(num_atoms, num_legs=4)

def _get_operations(coherent_control:CoherentControl):

    standard_operations : CoherentControl.StandardOperations = coherent_control.standard_operations(num_intermediate_states=0)    
    rotation    = standard_operations.power_pulse_on_specific_directions(power=1, indices=[0, 1, 2])
    p2_pulse    = standard_operations.power_pulse_on_specific_directions(power=2, indices=[0, 1])
 
    operations  = [
        rotation, p2_pulse, rotation, p2_pulse, rotation, p2_pulse, rotation, p2_pulse, rotation, p2_pulse, rotation, p2_pulse, rotation, p2_pulse, rotation, p2_pulse, rotation
    ]

    return operations

def _load_data(file_name:str, subfolder:str)->dict:
    return saveload.load(file_name, sub_folder=subfolder)


def _unpack_files_results(
    file_name:str, subfolder:str, coherent_control:CoherentControl, operations:list[Operation], initial_state:np.matrix
)->tuple[np.matrix, np.ndarray, float]:
    data = _load_data(file_name=file_name, subfolder=subfolder)
    score = -data["cost"]
    theta = data["theta"]
    if "state" in data:
        state = data["state"]
    else:
        state = coherent_control.custom_sequence(state=initial_state, theta=theta, operations=operations)
    return state, theta, score
    

def create_movie(
    state_type:StateType = StateType.Cat4,
    subfolder:str = "intermediate_results 2024.04.22_16.10.35 - good 24",
    num_atoms:int = 24,
    fps:int = 30,
    plot_target:bool = False,
    show_now:bool = False,
):
    
    # Start:
    print(f"Creating Movie...")
    _run_time_stamp=strings.time_stamp()

    coherent_control, initial_state, theta, operations, cost_function = _get_type_inputs(state_type=state_type, num_atoms=num_atoms, num_intermediate_states=0)

    ## File search:
    folder_full_path = saveload.DATA_FOLDER + os.sep + subfolder
    file_names = [file for file in os.listdir(folder_full_path)]

    # config:
    bloch_config = BlochSphereConfig(
        alpha_min=1, # no opacity
        resolution=150,
        viewing_angles=ViewingAngles(
            elev=-90,
            azim=+45
        )
    )

    def _create_matter_figure(state)->MatterStatePlot:
        return MatterStatePlot(initial_state=state, bloch_sphere_config=bloch_config, horizontal=False)   
    

    if plot_target:
        target = _get_target_state(num_atoms, state_type)
        target_plot = _create_matter_figure(target)
        target_plot.set_title("Target state")
        for ext in ["png", "tif"]:
            save_figure(target_plot.figure, file_name="optimization_target "+_run_time_stamp, extension=ext)

    def _get_results(filename): 
        return _unpack_files_results(
            filename, subfolder=subfolder, coherent_control=coherent_control, operations=operations, initial_state=initial_state
        )

    state, _, _ = _get_results(file_names[0])
    
    ## Animation elements:
    figure_object = _create_matter_figure(state) 
    video_recorder = VideoRecorder(fps=fps, temp_dir_name="optimization_frames "+_run_time_stamp)

    def _update_plot(state, score)->None:
        title = f"fidelity={score:.5f}"
        figure_object.update(state, title=title)
        video_recorder.capture(fig=figure_object.figure, duration=2)   
        if show_now:
            draw_now() 

    ## Iterate resutlts by result
    plot_every : int = 1
    plot_count : int = 0

    prog_bar = strings.ProgressBar(len(file_names), print_prefix="Capturing optimization movie:  ")
    for file_name in file_names:
        prog_bar.next()
        state, theta, score = _get_results(file_name)

        plot_count += 1
        if plot_count >= plot_every:
            _update_plot(state, score)
            plot_count = 0

    prog_bar.clear()

    video_recorder.write_video(name="optimization_movie "+_run_time_stamp)

    
    # Finish
    print("Done with movie.")



if __name__ == "__main__":
    create_movie()
    
    print("Done.")