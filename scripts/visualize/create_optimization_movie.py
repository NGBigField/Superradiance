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
from physics.famous_density_matrices import ground_state
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

# for enums:
from enum import Enum, auto

# for sleeping:
from time import sleep

# For writing results to file:
from csv import DictWriter

# For listing files in folder and navigating data:
import os

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
    final_state = coherent_control.custom_sequence(state=initial_state, theta=theta, operations=operations)
    return final_state, theta, score
    

def create_movie(
    num_atoms:int = 4,
    subfolder:str = "intermediate_results 2024.03.27_14.49.38",
    plot_target:bool = False,
    show_now:bool = False
):
    
    # Start:
    print(f"Creating Movie...")
    _run_time_stamp=strings.time_stamp()

    ## File search:
    folder_full_path = saveload.DATA_FOLDER + os.sep + subfolder
    file_names = [file for file in os.listdir(folder_full_path)]

    # config:
    bloch_config = BlochSphereConfig(
        alpha_min=0.1,
        resolution=150,
        viewing_angles=ViewingAngles(
            elev=-45,
            azim=+45
        )
    )

    # General variables:
    coherent_control = CoherentControl(num_atoms)
    operations = _get_operations(coherent_control)
    initial_state = ground_state(num_atoms)
    

    if plot_target:
        target = gkp_state(num_atoms, "square")
        target_plot = MatterStatePlot(
            initial_state=target,
            bloch_sphere_config=bloch_config
        )   
        save_figure(target_plot.figure, file_name="optimization_target "+_run_time_stamp)

    def _get_results(filename): 
        return _unpack_files_results(
            filename, subfolder=subfolder, coherent_control=coherent_control, operations=operations, initial_state=initial_state
        )

    state, _, _ = _get_results(file_names[0])
    
    ## Animation elements:
    figure_object = MatterStatePlot(initial_state=state, bloch_sphere_config=bloch_config) 
    video_recorder = VideoRecorder(fps=30, temp_dir_name="optimization_frames "+_run_time_stamp)

    def _update_plot(state, score)->None:
        title = f"fidelity={score:.5f}"
        figure_object.update(state, title=title)
        video_recorder.capture(fig=figure_object.figure)   
        if show_now:
            draw_now() 


    for file_name in file_names:
        state, theta, score = _get_results(file_name)
        _update_plot(state, score)

    video_recorder.write_video(name="optimization_movie "+_run_time_stamp)

    
    # Finish
    print("Done with movie.")



if __name__ == "__main__":
    create_movie()
    
    print("Done.")