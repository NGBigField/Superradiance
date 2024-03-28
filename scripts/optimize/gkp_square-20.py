# ==================================================================================== #
# |                                   Imports                                        | #
# ==================================================================================== #
if __name__ == "__main__":
    import pathlib, sys
    sys.path.append(
        str(pathlib.Path(__file__).parent.parent.parent)
    )

# Everyone needs numpy:
import numpy as np
from numpy import pi

# Use our utils:
from utils import strings

# For typing hints:
from typing import (
    Any,
    Tuple,
    List,
    Union,
    Dict,
    Final
)

# For coherent control
from algo.coherentcontrol import (
    CoherentControl,
    _DensityMatrixType,
    Operation,
)
        
# Import optimization options and code:
from algo.optimization import (
    LearnedResults,
    learn_custom_operation,
    learn_custom_operation_by_partial_repetitions,
    BaseParamType,
    FreeParam,
    FixedParam,
)

# For physical states:
from physics.gkp import get_gkp_cost_function
from physics.fock import Fock


# ==================================================================================== #
# |                                  Constants                                       | #
# ==================================================================================== #
DEFAULT_OPT_METHOD : Final[str] = "Nelder-Mead" #'SLSQP' # 'Nelder-Mead'

TOLERANCE : Final[float] = 1e-16  # 1e-12
MAX_NUM_ITERATION : Final[int] = int(1e6)  

T4_PARAM_INDEX : Final[int] = 5

# ==================================================================================== #
# |                                Inner Functions                                   | #
# ==================================================================================== #


# ==================================================================================== #
# |                                     main                                         | #
# ==================================================================================== #
   
def best_sequence_params(
    num_atoms:int, 
    /,*,
    num_intermediate_states:int=0    
)-> Tuple[
    List[BaseParamType],
    List[Operation]
]:
       
    # Define operations:
    coherent_control = CoherentControl(num_atoms=num_atoms)    
    standard_operations : CoherentControl.StandardOperations = coherent_control.standard_operations(num_intermediate_states=num_intermediate_states)
    
    rotation    = standard_operations.power_pulse_on_specific_directions(power=1, indices=[0, 1, 2])
    p2_pulse    = standard_operations.power_pulse_on_specific_directions(power=2, indices=[0, 1])
    stark_shift = standard_operations.stark_shift_and_rot()
        
    eps = 0.1    
        
    _rot_bounds   = lambda n : [(-pi-eps, pi+eps)]*n
    _p2_bounds    = lambda n : _rot_bounds(n) # [(None, None)]*n
    _stark_bounds = lambda n : [(None, None)]*n
    
    _rot_lock   = lambda n : [False]*n 
    _p2_lock    = lambda n : [False]*n
    _stark_lock = lambda n : [False]*n

    theta = [-0.001407818827965, 0.1218051907866897, -0.3771403547985771, 0.024935619382634, 0.2270067306882719,
              -0.019335708838509, 0.2510599878873806, -1.0608975361709554, 0.0024567612752524, 0.2156600716737191,
               0.031998891534794, 0.1560628844755161, -0.006424710131418, -0.0039155900133758, -0.066365152883213, 
                2.4500398390142815, -1.117489509540711, 1.408196183509574, -0.0483585560916238, -0.2901127398949547, 
                -2.2320056988442616, 2.2070671753828455, -1.5500987219879376, -1.1049676663529908, -0.2459302703358617, 
                0.0587932662988053, -2.59880224317601, -1.0736276867710544, 0.028771572678022, -0.1139445824659741, 
                -0.2676618670549614, -0.2193009148203622, 0.0562972222465117, 0.4029097063193843, 0.1678193666140538, 
                1.6503878341850395, -0.2424798467488802, -0.4116624135075853, -0.007748118131885, 0.1030546269865189, 
                -0.0113685049661158, -0.971750704798481, 1.0843898032444632
   ]

    operations  = [
        rotation, p2_pulse, rotation, p2_pulse, rotation, p2_pulse, rotation, p2_pulse, rotation, p2_pulse, rotation, p2_pulse, rotation, p2_pulse, rotation, p2_pulse, rotation
    ]

    
    num_operation_params : int = sum([op.num_params for op in operations])
    assert num_operation_params==len(theta)
    
    params_bound = []
    params_lock  = []
    for op in operations:
        n = op.num_params
        if op is rotation:
            params_bound += _rot_bounds(n)
            params_lock  += _rot_lock(n)
        elif op is stark_shift:
            params_bound += _stark_bounds(n)
            params_lock  += _stark_lock(n)
        elif op is p2_pulse:
            params_bound += _p2_bounds(n)
            params_lock  += _p2_lock(n)
        else:
            raise ValueError("Not an option")
    
    assert len(theta)==len(params_bound)==num_operation_params==len(params_lock)  
    param_config : List[BaseParamType] = []
    for i, (initial_value, bounds, is_locked) in enumerate(zip(theta, params_bound, params_lock)):        
        if is_locked:
            this_config = FixedParam(index=i, value=initial_value)
        else:
            this_config = FreeParam(index=i, initial_guess=initial_value, bounds=bounds, affiliation=None)   # type: ignore       
        param_config.append(this_config)
        
    
    return param_config, operations          
    
    
def main(
    num_atoms:int=20, 
    max_iter_per_attempt=1*int(1e5),
    tolerance=1e-17,
) -> LearnedResults:
        
    # Similar to previous method:
    cost_function = get_gkp_cost_function(num_atoms, form="square")
    initial_state = Fock.ground_state_density_matrix(num_atoms=num_atoms)
    
    # Params and operations:
    _, operations = best_sequence_params(num_atoms)

    results = learn_custom_operation(
        initial_state=initial_state, 
        cost_function=cost_function, 
        operations=operations, 
        max_iter=max_iter_per_attempt, 
        tolerance=tolerance,
        parameters_config=None,
        opt_method=DEFAULT_OPT_METHOD
    )


    print(results)
    return results

if __name__ == "__main__":
    results = main()
    print("Done.")