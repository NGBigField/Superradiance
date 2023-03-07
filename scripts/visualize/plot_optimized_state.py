# ==================================================================================== #
#| Imports:                                         
# ==================================================================================== #

if __name__ == "__main__":
    import sys, pathlib
    sys.path.append(
        pathlib.Path(__file__).parent.parent.parent.__str__()
    )

# Import useful types:
from algo.optimization import BaseParamType
from algo.coherentcontrol import Operation

# Basic algo mechanisms:
from algo.coherentcontrol import CoherentControl
from physics.famous_density_matrices import ground_state

# for numerics:
import numpy as np

# Our best optimized results:    
from scripts.optimize.cat4_thin  import best_sequence_params as cat4_params
from scripts.optimize.gkp_hex    import best_sequence_params as gkp_hex_params
from scripts.optimize.gkp_square import best_sequence_params as gkp_square_params

# for plotting:
from utils.visuals import plot_matter_state, plot_light_wigner

# for enums:
from enum import Enum, auto



# ==================================================================================== #
#| Helper types:
# ==================================================================================== #

class Res(Enum):
    HexGKP = auto()
    SquareGKP = auto()
    Cat4 = auto()


# ==================================================================================== #
#| Inner Functions:
# ==================================================================================== #

def _get_best_params(
    type_:Res, 
    num_atoms:int
) -> tuple[
    list[BaseParamType],
    list[Operation]
]:
    if type_ is Res.HexGKP:
        return gkp_hex_params(num_atoms)
    elif type_ is Res.SquareGKP:
        return gkp_square_params(num_atoms)
    elif type_ is Res.Cat4:
        return cat4_params(num_atoms)
    else:
        raise ValueError(f"Not an option '{type_}'")


def _get_best_final_result(type_:Res, num_atoms:int) -> np.matrix:
    # Get all needed data:
    params, operations = _get_best_params(type_, num_atoms)
    initial_state = ground_state(num_atoms=num_atoms)
    coherent_control = CoherentControl(num_atoms=num_atoms)
    
    # derive theta:
    theta = [param.get_value() for param in params]
    
    # create state:
    final_state = coherent_control.custom_sequence(state=initial_state, theta=theta, operations=operations)
    return final_state
    

# ==================================================================================== #
#| Declared Functions:
# ==================================================================================== #







## Main:
def main(
    type_:Res = Res.SquareGKP,
    num_atoms:int = 40
):
    # get
    final_state = _get_best_final_result(type_=type_, num_atoms=num_atoms)
    
    # plot
    plot_light_wigner(final_state)
    
    # Done:
    print("Done.")
    

if __name__ == "__main__":
    main()