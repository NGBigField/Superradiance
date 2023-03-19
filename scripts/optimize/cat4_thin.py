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

# For operations and cost functions:
from physics.fock import Fock, cat_state
from algo import metrics



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
    #     +1.6911753538276657 , +0.4165367678990034 , +1.1596610642766465 , +0.4970010986390708 , +1.1626455201688501 , 
    #     +0 , +0 , +0 , +0 , +0 , 
    #     -0.8365536257598889 , -1.0001921078914235 , +1.3845575396713630 
    # ]
    # theta = [
    #     +0.4318330313098309 , +1.6226884800739532 , -0.5184544170040373 , +0.4904292546661787 , +1.1579761103653765 , 
    #     +0.2711072924647956 , -1.1455548573063417 , -0.0110646563013583 , +0.0058963857195526 , +0.0093821994475182 , 
    #     -0.9014452124841090 , -1.6577610967809480 , +1.8807033704653549         
    # ]
    # theta = [
    #     +0.3658449972290249 , +0.9257611370387263 , -1.6882308749463719 , +0.4376024809598818 , +1.1897942064568279 , 
    #     +1.1670054840863089 , -1.3422791867903432 , +0.4158322335275012 , +0.0188350187791549 , +0.0213576999852772 , 
    #     -0.0528461850923196 , -1.9429029841060719 , +1.2938917481256631       
    # ]
    theta = [
        +0.3887435406908402 , +0.9104233504317537 , -1.6882308749463719 , +0.4379801564966822 , +1.1907636404558120 ,
        0.0 , 0.0 , 0.0 , 0.0 , 0.0 , 
        +1.1655035689193887 , -1.3427208539321607 , +0.4163584988465759 , +0.0187488498052232 , +0.0211625884315972 ,
        -0.0541798806370485 , -1.9406363397783859 , +1.2944312375423341
    ]
    
    operations  = [
        rotation, p2_pulse, rotation, p2_pulse, rotation, p2_pulse, rotation
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
    num_moments:int=40, 
    num_total_attempts:int=2000, 
    
    num_runs_per_attempt:int=4*int(1e3), 
    max_error_per_attempt:Optional[float]=1e-12,
    num_free_params:int|None=9,
    sigma:float=0.004
) -> LearnedResults:
    
    # Define target:
    target_4legged_cat_state = cat_state(num_atoms=num_moments, alpha=3, num_legs=4).to_density_matrix()
    initial_state = Fock.ground_state_density_matrix(num_atoms=num_moments)
    def cost_function(final_state:_DensityMatrixType) -> float : 
        return -1 * metrics.fidelity(final_state, target_4legged_cat_state)  
    
    # Define operations:
    param_config, operations = best_sequence_params(num_moments)

    best_result = learn_custom_operation_by_partial_repetitions(
        # Mandatory Inputs:
        initial_state=initial_state,
        cost_function=cost_function,
        operations=operations,
        initial_params=param_config,
        # Heuristic Params:
        max_iter_per_attempt=num_runs_per_attempt,
        max_error_per_attempt=max_error_per_attempt,
        num_free_params=num_free_params,
        sigma=sigma,
        num_attempts=num_total_attempts,
        log_name="4-Cat-thin "+strings.time_stamp()
    )

    ## Finish:
    sounds.ascend()
    print(best_result)
    return best_result

if __name__ == "__main__":
    results = main()
    print("Done.")