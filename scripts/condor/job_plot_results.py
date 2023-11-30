import pathlib, sys

if __name__ == "__main__":
    sys.path.append(
        str(pathlib.Path(__file__).parent.parent.parent)
    )

from scripts.visualize.plot_optimized_state import plot_result, StateType

from typing import Any
from time import sleep


def main(variation:int=0, num_grahpic_points=2000) -> dict[str, Any]:
    
    ## Constants:
    num_atoms=40

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
    score = plot_result(state_type=state_type, create_movie=False, num_graphics_points=num_grahpic_points)

    return dict(
        variation = name,
        seed = -1,
        score = score,
        theta = -1
    )


if __name__ == "__main__":
    main()

