
if __name__ == "__main__":
    import sys, pathlib
    sys.path.append(
        pathlib.Path(__file__).parent.parent.__str__()
    )

from physics.emitted_light_approx import main as _calc_emitted_light_from
import numpy as np
from utils import saveload

from scripts.visualize.plot_optimized_state import _get_type_inputs, _get_movie_config, StateType



def _get_matter_state(state_type:StateType):
    # get basic
    coherent_control, initial_state, theta, operations, cost_function = _get_type_inputs(state_type=state_type, num_atoms=40, num_intermediate_states=0)
    movie_config = _get_movie_config(create_movie=False, num_transition_frames=0, state_type=state_type)    
    
    # create state:
    final_state = coherent_control.custom_sequence(state=initial_state, theta=theta, operations=operations, movie_config=movie_config)

    return final_state



def calc_emitted_light(state_type:StateType, time_resolution:int=200) -> np.matrix:
    matter_state : np.matrix = _get_matter_state(state_type)
    # Basic info:
    sub_folder = "Emitted-Light"
    file_name = state_type.name + f" time_res_{time_resolution}"
    # Get or calc:
    emitted_light_state = _calc_emitted_light_from(matter_state, t_final=0.1, time_resolution=time_resolution)
    saveload.save(emitted_light_state, name=file_name, sub_folder=sub_folder)
    # Return:
    return emitted_light_state #type: ignore









if __name__ == "__main__":
    calc_emitted_light(StateType.GKPSquare,  time_resolution=50)
    
    print("Done.")