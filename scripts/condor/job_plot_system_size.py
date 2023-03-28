import pathlib, sys

if __name__ == "__main__":
    sys.path.append(
        str(pathlib.Path(__file__).parent.parent.parent)
    )

from scripts.visualize.plot_optimized_state import plot_result, StateType
from typing import Any

NUM_OPTIONS = 11 


def main(variation:int=0, seed:int=0) -> dict[str, Any]:
    
    ## Constants:

    ## derive input:
    lower, upper = 40-NUM_OPTIONS+1, 40+NUM_OPTIONS+1
    num_atoms_options = list(range(lower, upper, 2))
    index = seed
    num_atoms = num_atoms_options[index]

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
    

    ## Create movie:
    score = plot_result(state_type=state_type, num_atoms=num_atoms)


    return dict(
        variation = name,
        seed = seed,
        score = score,
        theta = -1
    )


if __name__ == "__main__":
    main()

