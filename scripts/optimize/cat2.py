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
    #     +3.2415926535897932 , +0.1828973771603937 , -0.3865228829159012 , +1.0576654516530777 , +1.9488788559183661 , 
    #     +3.2415926535897932 , -0.2227421037234675 , +2.2617422360192738 , +1.1431510776419032 , +0.9743504103718079 , 
    #     -0.4536053424551550 , +1.7575897551939796 , +2.6764489388084356 
    # ]
    # theta = [
    #     +1.6368658958523632 , +1.7027807751964548 , +1.6306314723134756 , +2.0693531144350454 , +0.9749253531352750 , 
    #     +2.2598187757205923 , +1.4521477435029744 , +1.4746351015538939 , +2.4653392641318455 , +3.2415926535897932 , 
    #     +0.5496666717820531 , +0.5376274709294500 , +1.7684380007018543 
    # ]    
    # theta = [
    #     +2.2884243619726465 , +1.3277700706962539 , +2.0870455334233835 , +1.9391807444535574 , +0.7816679973293168 ,
    #     +2.2504873168046138 , +1.3828220613581861 , +1.4841829260109276 , +2.4758211483630150 , +3.2401940268953995 ,
    #     +0.5455567960592630 , +0.4546373693341874 , +1.6935620833025031        
    # ]
    # theta = [
    #     +2.5066248308223518 , +1.1532063519469733 , +2.1315549524903052 , +1.9395278536116698 , +0.7820697532208565 , 
    #     +2.2494831260803050 , +1.3851850686407987 , +1.4840158399534116 , +2.4757063126764716 , +3.2400495813112133 , 
    #     +0.5461308198007867 , +0.4539199743584972 , +1.6939129421385699         
    # ]
    # theta = [
    #     2.2188, 1.4497, 1.9767, 1.8809, 0.6809, 2.4522, 1.6573, 1.3389, 2.4622, 3.2328, 0.5011, 0.584,  1.7406
    # ] # fidelity = 0.937194595494278
    theta = [
        2.1345128168028236, 1.5089496740531838, 1.9558262087960614, 1.877331551981269, 0.6773288810263087, 
        2.4113269886279887, 1.6645958630565674, 1.354121372951348, 2.4640768892843603, 3.231830426074341, 
        0.4955335029252635, 0.5774086130028968, 1.717606862784992
    ]  # fidelity = 0.938   - 2 steps
    
    operations  = [
        rotation, p2_pulse, rotation, p2_pulse, rotation
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
    max_iter_per_attempt:int=8*int(1e3), 
    max_error_per_attempt:Optional[float]=1e-17,
    num_free_params:int|None=10,
    sigma:float=0.005,
    initial_sigma:float=0.01
) -> LearnedResults:
    
    # Define target:
    initial_state = ground_state(num_atoms=num_atoms)    
    cost_function = fidelity_to_cat(num_atoms=num_atoms, num_legs=2)
    
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