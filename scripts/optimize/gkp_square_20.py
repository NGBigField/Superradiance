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

    theta = [
        +0.5980036599632439 , -0.1272247236387578 , +1.1896494831546907 , -0.1086032932919804 , -2.2953092643185888 ,
        -1.8617020354653471 , +0.5630358308916805 , -1.2099077014694615 , +2.2453289581088978 , +0.8414195064267818 ,
        -2.9383157387461969 , +3.0587168344861047 , -1.3648178953925925 , +1.9620437674916786 , -0.1045266634041619 ,
        +0.2973354809029606 , +0.9975013749355339 , +2.6717692218532383 , +0.2127916149772886 , -2.5896039514697993 ,
        +1.9436565549630005 , -1.4840439033466022 , +0.0946212533745333 , +1.6557916307050267 , -1.1818982914029850 ,
        +0.7859795491348851 , -0.1951557764775538 , +2.2973614164810341 , +1.8870203407413224 , -2.4859233455781196 ,
        -0.5801950917352418 , +0.0547263384814179 , +2.3175478860531187 , +1.4963954088964915 , -2.8194138821814940 ,
        -1.4245739509856388 , +2.2759855791748382 , +2.4253328971668018
   ]

    operations  = [
        rotation, p2_pulse, rotation, p2_pulse, rotation, p2_pulse, rotation, p2_pulse, rotation, p2_pulse, rotation, p2_pulse, rotation, p2_pulse, rotation
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
    save_intermediate_results:bool=True
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
        opt_method=DEFAULT_OPT_METHOD,
        save_intermediate_results=save_intermediate_results
    )


    print(results)
    return results

if __name__ == "__main__":
    results = main()
    print("Done.")