import pathlib, sys

if __name__ == "__main__":
    sys.path.append(
        str(pathlib.Path(__file__).parent.parent.parent)
    )

from scripts.visualize.plot_optimized_state import create_movie, StateType

from typing import Any


def main(variation:int=3, num_transition_frames:int=2) -> dict[str, Any]:
    
    ## Constants:
    num_atoms=40

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
    
    score = create_movie(state_type=state_type, num_atoms=num_atoms, num_transition_frames=num_transition_frames)


    return dict(
        variation = name,
        seed = -1,
        score = score,
        theta = -1
    )


if __name__ == "__main__":
    main()

