# ==================================================================================== #
# |                                   Imports                                        | #
# ==================================================================================== #
if __name__ == "__main__":
    import pathlib, sys
    sys.path.append(
        pathlib.Path(__file__).parent.parent.parent.__str__()
    )
    
# Everyone needs numpy:
import numpy as np
from numpy import pi

# For typing hints:
from typing import Optional, Tuple, List

# import our helper modules
from utils import sounds, strings

# For coherent control
from algo.coherentcontrol import (
    CoherentControl,
    _DensityMatrixType,
)

# Import optimization options and code:
from algo.optimization import (
    LearnedResults,
    learn_custom_operation_by_partial_repetitions,
    FixedParam, 
    FreeParam,
    BaseParamType,
    Operation
)

# Common states and cost functions:
from physics.famous_density_matrices import ground_state
from algo.common_cost_functions import fidelity_to_cat



# ==================================================================================== #
# |                                  Constants                                       | #
# ==================================================================================== #

# ==================================================================================== #
# |                                Inner Functions                                   | #
# ==================================================================================== #


def best_sequence_params(
    num_atoms:int,
    /,*,
    num_intermediate_states:int=0
)-> Tuple[
    List[BaseParamType],
    List[Operation]
]:
    
    coherent_control = CoherentControl(num_atoms=num_atoms)    
    standard_operations : CoherentControl.StandardOperations  = coherent_control.standard_operations(num_intermediate_states=num_intermediate_states)
    
    rotation  = standard_operations.power_pulse_on_specific_directions(power=1, indices=[0, 1, 2])
    squeezing = standard_operations.power_pulse_on_specific_directions(power=2, indices=[0, 1])
        
    eps = 0.1    
        
    _rot_bounds   = lambda n : [(-pi-eps, pi+eps)]*n
    _p2_bounds    = lambda n : _rot_bounds(n) # [(None, None)]*n
    
    _rot_lock   = lambda n : [False]*n 
    _p2_lock    = lambda n : [False]*n
   
    # theta = [
    #     0.00, 0.00, 0.00, 
    #     0.00, 0.00, 
    #     0.00, 0.00, 0.00
    # ]  # 1 step - ???? fidelity
    theta = [
        +0.0458066476464166 , +0.3234602214212449 , -0.6047948670764272 , -0.0604787266299985 , +0.0849994457593420 , 
        -0.5742503915384303 , -1.3959327732218383 , +0.5420776618416453
    ]  # 1 step - 0.27 fidelity

    operations  = [
        rotation, squeezing, rotation
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
        elif op is squeezing:
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



# ==================================================================================== #
#|                                    Main                                            |#
# ==================================================================================== #

    
def main(
    num_atoms:int=40, 
    num_total_attempts:int=1000, 
    max_iter_per_attempt:int=2*int(1e3), 
    max_error_per_attempt:Optional[float]=1e-10,
    num_free_params:int|None=6,
    sigma:float=0.05,
    initial_sigma:float=0.00
) -> LearnedResults:
    
    # Define target:
    initial_state = ground_state(num_atoms=num_atoms)    
    cost_function = fidelity_to_cat(num_atoms=num_atoms, num_legs=4, phase=np.pi/4)
    
    # Define operations:
    param_config, operations = best_sequence_params(num_atoms)

    best_result = learn_custom_operation_by_partial_repetitions(
        # Mandatory Inputs:
        initial_state=initial_state,
        cost_function=cost_function,
        operations=operations,
        initial_params=param_config,
        # Heuristic Params:
        initial_sigma=initial_sigma,
        max_iter_per_attempt=max_iter_per_attempt,
        max_error_per_attempt=max_error_per_attempt,
        num_free_params=num_free_params,
        sigma=sigma,
        num_attempts=num_total_attempts,
        log_name="4-Cat-i "+strings.time_stamp()
    )

    ## Finish:
    sounds.ascend()
    print(best_result)
    return best_result

if __name__ == "__main__":
    results = main()
    print("Done.")