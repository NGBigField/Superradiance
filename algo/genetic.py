# ==================================================================================== #     
# |                                   Imports                                        | #
# ==================================================================================== #

# Everyone needs numpy:
import numpy as np
from numpy import pi

# For typing hints:
from typing import (
    Callable,
)

# import our helper modules
from utils import (
    strings,
    sounds,
    errors,
)

# For coherent control
from coherentcontrol import (
    CoherentControl,
    _DensityMatrixType,
    Operation,
)
        
# For defining cost functions:
import metrics

# For operations:
import coherentcontrol
from fock import Fock, cat_state
from gkp import goal_gkp_state

# For managing saved data:
from utils.saved_data_manager import NOON_DATA, exist_saved_noon, get_saved_noon, save_noon

# For translation between pulses-param-space and optimization-parameter-space:
from optimization_and_operations import pair_custom_operations_and_opt_params_to_op_params, free_all_params

# For optimization:
from geneticalgorithm import geneticalgorithm

# ==================================================================================== #
# |                                Inner Functions                                   | #
# ==================================================================================== #
def get_gkp_cost_function(num_moments:int)->Callable[[_DensityMatrixType], float]:
    # Define target:
    # target_4legged_cat_state = cat_state(num_moments=num_moments, alpha=3, num_legs=4).to_density_matrix()
    taget_state = goal_gkp_state(num_moments)
    # visuals.plot_matter_state(target_4legged_cat_state, block_sphere_resolution=200)
    def cost_function(final_state:_DensityMatrixType) -> float : 
        return -1 * metrics.fidelity(final_state, taget_state)       

    return cost_function




# ==================================================================================== #
# |                                     main                                         | #
# ==================================================================================== #

 
def opt_gkp_by_gentics_algo(
    num_moments:int=40, 
    num_attempts:int=2000, 
    num_free_params:int|None=20
):
    
    # Constants:
    num_transition_frames:int=0
    
    
    
    # Define operations:
    coherent_control = CoherentControl(num_moments=num_moments)    
    standard_operations : CoherentControl.StandardOperations  = coherent_control.standard_operations(num_intermediate_states=num_transition_frames)
    
    
    rotation    = standard_operations.power_pulse_on_specific_directions(power=1, indices=[0, 1, 2])
    p2_pulse    = standard_operations.power_pulse_on_specific_directions(power=2, indices=[0, 1])
    stark_shift = standard_operations.stark_shift_and_rot()
        
    eps = 0.1    
        
    _bounds   = lambda n : [(-pi-eps, pi+eps)]*n

    operations  = [
        rotation, p2_pulse, rotation, p2_pulse, rotation, p2_pulse, rotation, p2_pulse, rotation, p2_pulse, rotation, p2_pulse, rotation, p2_pulse, rotation,  p2_pulse, rotation, p2_pulse, rotation
    ] 
    
    # Similar to previous method:
    gkp_cost_function = get_gkp_cost_function(num_moments)
    initial_state = Fock.ground_state_density_matrix(num_moments=num_moments)
    def total_cost_function(theta:np.ndarray) -> float : 
        final_state = coherent_control.custom_sequence(initial_state, theta=theta, operations=operations )
        cost = gkp_cost_function(final_state)
        return cost
    
    num_params = sum([op.num_params for op in operations])  
    
    ## Run:
    variable_boundaries = []
    for op in operations:
        variable_boundaries += _bounds(op.num_params)
    variable_boundaries = np.array(variable_boundaries)

    model = geneticalgorithm(function=total_cost_function, dimension=num_params, variable_type='real', variable_boundaries=variable_boundaries)

    model.run()
    sounds.ascend()
    
    try:
        print(model)
    except Exception as e:
        errors.print_traceback(e)
    
    try:        
        print(model.report)
        print(model.ouput_dict)
    except Exception as e:
        errors.print_traceback(e)
    
    
    print("Done")

if __name__ == "__main__":
    results = opt_gkp_by_gentics_algo()
    print("Done.")
    
    