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
from utils import sounds, strings, visuals

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
from physics.famous_density_matrices import cat_state, ground_state
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
   

    # theta = [
    #     0.0005128168028236, 0.0089496740531838, 0.0008262087960614, 0.000331551981269, 0.0773288810263087, 
    #     0.0005335029252635, 0.0004086130028968, 0.000606862784992
    # ]  # fidelity = ?????  - 1 steps
    # theta = [
    #     +2.2634889386413555 , +0.0191982536001193 , -1.8928371986018913 , -0.0333016826990670 , +0.2205461238230759 ,
    #     -0.0210681193831718 , +0.4497206129835951 , +0.1635441051534410
    # ] # fidelity = 0.5  - 1 steps
    # theta = [
    #     -0.5760774336082597 , +0.1323903179629796 , -3.2415926535897932 , 
    #     +0.0393305489457032 , +0.8003328339843916 , 
    #     -0.1019899724435195 , +0.7026257860191545 , -0.0335225940181096
    # ] # fidelity = 0.695  - 1 steps
    # theta = [
    #     +1.0104922822465972 , -0.3081407181418243 , +0.3957481641184154 , +1.6107168713010418 , +1.5722357760249195 , 
    #     +0.2827102621043437 , -0.1090679797134472 , +1.2558451565627515
    # ] #  fidelity = 0.743  - 1 steps
    theta = [
        +1.6672585088573388 , +0.7966649375214807 , +3.2415926535897932 , +1.5714319595349933 , +1.5701701865717275 , 
        -0.0000111217866419 , -0.0000015622659345 , +2.3594798466050158
    ] # fidelity 0.9918 - 1 step
    
    
    operations  = [
        rotation, p2_pulse, rotation
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



# ==================================================================================== #
#|                                    Main                                            |#
# ==================================================================================== #

    
def main(
    num_atoms:int=40, 
    num_total_attempts:int=2000, 
    max_iter_per_attempt:int=2*int(1e3), 
    max_error_per_attempt:Optional[float]=1e-20,
    num_free_params:int|None=7,
    sigma:float=0.005,
    initial_sigma:float=0.0000
) -> LearnedResults:
    
    # Define target:
    initial_state = ground_state(num_atoms=num_atoms)    
    cost_function = fidelity_to_cat(num_atoms=num_atoms, num_legs=2, phase=np.pi/2)
    
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
        log_name="2-Cat"+strings.time_stamp()
    )

    ## Finish:
    sounds.ascend()
    print(best_result)
    return best_result

if __name__ == "__main__":
    results = main()
    print("Done.")