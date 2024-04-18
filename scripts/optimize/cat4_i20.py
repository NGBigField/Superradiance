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
from typing import Optional, Tuple, List, Final

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
    learn_custom_operation,
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
DEFAULT_OPT_METHOD : Final[str] = "Nelder-Mead" #'SLSQP' # 'Nelder-Mead'

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
   
    theta = [
        +0.8937567499106599 , +3.2085033698137830 , -2.3242661423839071 , +0.0036751816770657 , -0.7836001773757240 , 
        +0.8937567499106599 , +3.2085033698137830 , -2.3242661423839071 , +0.0036751816770657 , -0.7836001773757240 , 
        +0.8937567499106599 , +3.2085033698137830 , -2.3242661423839071 , +0.0036751816770657 , -0.7836001773757240 , 
        +2.6065083924231915 , +2.2505047554207338 , -2.4740789195081394        
    ] 

    operations  = [
        rotation, squeezing, 
        rotation, squeezing, 
        rotation, squeezing, 
        rotation
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
    num_atoms:int=20, 
    max_iter_per_attempt=1*int(1e5),
    tolerance=1e-17,
    save_intermediate_results:bool=True
) -> LearnedResults:
    
    # Define target:
    initial_state = ground_state(num_atoms=num_atoms)    
    cost_function = fidelity_to_cat(num_atoms=num_atoms, num_legs=4, phase=np.pi/4)
    
    # Define operations:
    param_config, operations = best_sequence_params(num_atoms)

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


    ## Finish:
    sounds.ascend()
    print(results)
    return results

if __name__ == "__main__":
    results = main()
    print("Done.")