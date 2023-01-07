
# ==================================================================================== #
#|                                    Imports                                         |#
# ==================================================================================== #

# Everyone needs numpy:
import numpy as np
from numpy import array

# our utilities:
from utils import (
    visuals,
)

# For type hinting:
from typing import (
    List,
    Union,
    Tuple,
    Generator,
)

# metrics:
from metrics import fidelity

# Operations
from coherentcontrol import Operation, _deal_costum_params, CoherentControl, _DensityMatrixType

# Optimization
from optimization import OptimizationParams, ParamConfigBase, _deal_params_config, _positive_indices_from_operations

# Fock:
from fock import Fock, coherent_state, cat_state

# ==================================================================================== #
#|                               Declared Functions                                   |#
# ==================================================================================== #


def pair_custom_operations_and_opt_params_to_op_params(
    operations: List[Operation],
    opt_theta: Union[List[float], np.ndarray],
    parameters_config
)->Generator[Tuple[Operation, List[float]], None, None]:
    
    num_operation_params = sum([op.num_params for op in operations])
    positive_indices = _positive_indices_from_operations(operations)
    param_config : OptimizationParams = _deal_params_config(num_operation_params, positive_indices, parameters_config)
    operation_theta = param_config.optimization_theta_to_operations_params(opt_theta)
    
    all_ops_params : List[List[float]] = _deal_costum_params(operations, operation_theta)
    for params, operation in zip(all_ops_params, operations):
        yield operation, params



# ==================================================================================== #
#|                                       Main                                         |#
# ==================================================================================== #




def main_test():
    
    num_moments = 40
    
    opt_theta = array(
        [   3.03467614,    0.93387172,  -10.00699257,   -0.72388404,
            0.13744785,    2.11175319,    0.18788428, -118.69022356,
            -1.50210956,    2.02098048,   -0.21569011,   -2.9236711 ,
            3.01919738,    3.14159265,   -0.32642685,   -0.87976521,
            -0.83782409])
    
    from main import _common_4_legged_search_inputs
    initial_state, cost_function, cat4_creation_operations, param_config = _common_4_legged_search_inputs(num_moments)
    
    theta = []
    for operation, oper_params in  pair_custom_operations_and_opt_params_to_op_params(cat4_creation_operations, opt_theta, param_config):
        print(operation.get_string(oper_params))
        theta.extend(oper_params)
        
    target_4legged_cat_state = cat_state(num_moments=num_moments, alpha=3, num_legs=4).to_density_matrix()
    def _score_str_func(rho:_DensityMatrixType)->str:
        fidel = fidelity(rho, target_4legged_cat_state)
        return f"fidelity={fidel}"
    
    
    operations = cat4_creation_operations
    
    coherent_control = CoherentControl(num_moments=num_moments)
    movie_config = CoherentControl.MovieConfig(
        active=False,
        show_now=True,
        num_transition_frames=2,
        num_freeze_frames=2,
        bloch_sphere_resolution=10,
        score_str_func=_score_str_func
    )
    final_state = coherent_control.custom_sequence(initial_state, theta=theta, operations=operations, movie_config=movie_config)
    print(final_state)
    print(_score_str_func(final_state))
    visuals.plot_matter_state(final_state)
    print("Done.")
    
        


if __name__ == "__main__":
    main_test()
    print("Done.")