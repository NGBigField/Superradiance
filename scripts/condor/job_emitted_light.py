import pathlib, sys

if __name__ == "__main__":
    sys.path.append(
        str(pathlib.Path(__file__).parent.parent.parent)
    )

from scripts.visualize.plot_optimized_state import StateType, _get_type_inputs
from scripts.emitted_light import calc_emitted_light

from typing import Any

from time import sleep


def main(variation:int=0, time_resolution:int=200) -> dict[str, Any]:
    
    ## Constants:
    num_atoms=40
    num_transition_frames:int=0

    ## Choose method:
    if variation==0:
        state_type = StateType.GKPSquare
        name = "square_gkp"
    elif variation==1:
        state_type = StateType.GKPHex
        name = "hex_gkp"
    elif variation==2:
        state_type = StateType.Cat4
        name = "cat4"
    elif variation==3:
        state_type = StateType.Cat2
        name = "cat2"
    else:
        raise ValueError(f"Not a supported variation: {variation}")
    
    ## Sleep:
    sleep(variation)

    ## Create movie:
    light_state = calc_emitted_light(state_type=state_type, time_resolution=time_resolution)
    coherent_control, initial_state, theta, operations, cost_function = _get_type_inputs(state_type=state_type, num_atoms=num_atoms, num_intermediate_states=num_transition_frames)
    fidelity = -cost_function(light_state) 

    return dict(
        variation = name,
        seed = -1,
        score = fidelity,
        theta = -1,
        time_resolution = time_resolution
    )


if __name__ == "__main__":
    main()

