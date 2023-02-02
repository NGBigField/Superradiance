# ==================================================================================== #
# |                                   Imports                                        | #
# ==================================================================================== #

# Everyone needs numpy:
import numpy as np
from numpy import pi

# import our helper modules
from utils import (
    assertions,
    numpy_tools as np_utils,
    visuals,
    saveload,
    strings,
    sounds,
    lists,
    decorators,
    errors,
)

# for type annotating:
from typing import List 
from types import SimpleNamespace

from copy import deepcopy

# For defining coherent states:
from fock import Fock


# For coherent control
from coherentcontrol import CoherentControl, _DensityMatrixType

# for plotting
import matplotlib.pyplot as plt  # for plotting test results:

# For optimizations:
from optimization import learn_custom_operation, LearnedResults, minimize, learn_single_op, FreeParam

from gkp import get_gkp_cost_function

# For optimization:
from geneticalgorithm import geneticalgorithm

# ==================================================================================== #
#|                                helper functions                                    |#
# ==================================================================================== #

def _get_final_state(
    ground_state:_DensityMatrixType, coherent_control:CoherentControl,
    x1, x2
):
    # Basic pulses:
    Sy = coherent_control.s_pulses.Sy
    Sy2 = Sy@Sy
    Sz = coherent_control.s_pulses.Sz

    # Act with pulses:
    rho = ground_state
    rho = coherent_control.pulse_on_state(rho, x=x1, power=2) 
    rho, z1 = learn_single_op(rho, Sz, Sy2)

    z2 = pi/2

    rho = coherent_control.pulse_on_state(rho, x=x2, power=1)
    rho = coherent_control.pulse_on_state(rho, z=z2, power=2)

    rho = coherent_control.pulse_on_state(rho, x=x2, power=1)
    rho = coherent_control.pulse_on_state(rho, z=z2, power=2)

    visuals.plot_light_wigner(rho)
    visuals.draw_now()

    return rho

# ==================================================================================== #
#|                                   main tests                                       |#
# ==================================================================================== #
    
def _example_gkp2(
    num_moments = 40
):

    # Define the basics:
    ground_state = Fock.ground_state_density_matrix(num_moments=num_moments)    
    coherent_control = CoherentControl(num_moments=num_moments)
    final_state = lambda x1, x2 : _get_final_state(ground_state, coherent_control, x1, x2)

    # Derive size-specific variables:
    if num_moments==20:
        x1 = 0.02
        x2 = 0.8
    elif num_moments==40:
        x1 = 0.09
        x2 = 0.2
    elif num_moments==100:
        x1 = 0.02
        x2 = 0.4
    else:
        raise ValueError(f"This number is not supported. num_moments={num_moments}")

    # Act with Pulses:
    x1, x2 = 0.072, 0.3
    final_state(x1, x2)
    print("Finished.")
    


def _alexeys_recipe(num_moments:int=100):
    # Define the basics:
    ground_state = Fock.ground_state_density_matrix(num_moments=num_moments)    
    coherent_control = CoherentControl(num_moments=num_moments)

    # Learned parameters:
    
    N = num_moments
    r = 1.1513
    theta = 0
    R = np.sqrt( (np.cosh(4*r)) / (2 * (N**2) ) )
    phi = np.arcsin( (-2*N*R) / np.sqrt(4 * (N**4) * (R**4) - 1) )

    x2 = 7.4
    z2 = pi/2

    # Act with pulses:
    rho = ground_state

    rho = coherent_control.pulse_on_state(rho, x=R,    power=2)
    rho = coherent_control.pulse_on_state(rho, z=-phi, power=1) 

    rho = coherent_control.pulse_on_state(rho, x=x2, power=1)
    rho = coherent_control.pulse_on_state(rho, z=z2, power=2)

    rho = coherent_control.pulse_on_state(rho, x=x2, power=1)
    rho = coherent_control.pulse_on_state(rho, z=z2, power=2)

    # Plot:
    visuals.plot_light_wigner(rho)
    visuals.draw_now()
    print("Done.")
    

def learning(
    num_moments:int=40,
    num_attempts:int=100,
    max_iter=int(1e4),
    sigma:float=0.05
):

   # Similar to previous method:
    cost_function = get_gkp_cost_function(num_moments)
    initial_state = Fock.ground_state_density_matrix(num_moments=num_moments)
    
    # Define operations:
    coherent_control = CoherentControl(num_moments=num_moments)    
    standard_operations : CoherentControl.StandardOperations  = coherent_control.standard_operations(num_intermediate_states=0)
    operations = [
        standard_operations.power_pulse_on_specific_directions(power=1, indices=[0,1,2]),
        standard_operations.power_pulse_on_specific_directions(power=2, indices=[0]),
        standard_operations.power_pulse_on_specific_directions(power=1, indices=[2]),
        standard_operations.power_pulse_on_specific_directions(power=1, indices=[0]),
        standard_operations.power_pulse_on_specific_directions(power=2, indices=[2]),
        standard_operations.power_pulse_on_specific_directions(power=1, indices=[0]),
        standard_operations.power_pulse_on_specific_directions(power=2, indices=[2]),
        standard_operations.power_pulse_on_specific_directions(power=1, indices=[0,1,2])
    ]

    ## Params:
    N = num_moments
    r = 1.1513
    theta = 0
    R = np.sqrt( (np.cosh(4*r)) / (2 * (N**2) ) )
    phi = np.arcsin( (-2*N*R) / np.sqrt(4 * (N**4) * (R**4) - 1) )
    x2 = 7.4
    z2 = pi/2
    eps = 0.1  
    _bounds = lambda: (-pi-eps, pi+eps)


    initial_values = [0.0, 0.0, 0.0, R, -phi, x2, z2, x2, z2, 0.0, 0.0, 0.0]





    best_result = SimpleNamespace(score=np.inf)

    theta = deepcopy(initial_values)
    for i in range(num_attempts):

        param_config : List[FreeParam] = []
        for i, initial_value in enumerate(theta):
            param = FreeParam(
                index=i, 
                initial_guess=initial_value+np.random.normal(1)*sigma, 
                bounds=_bounds(), 
                affiliation=None
            )   # type: ignore       
            param_config.append(param)
            
    
        result = learn_custom_operation(
            num_moments=num_moments, 
            initial_state=initial_state, 
            cost_function=cost_function, 
            operations=operations, 
            max_iter=max_iter, 
            parameters_config=param_config
        )

        if result.score < best_result.score:
            print("Best result!")
            best_result = result
            theta = result.operation_params

        print(result)
        print(" ")

    sounds.ascend()
    print(best_result)
    return best_result



def learning_by_genetics(
    num_moments:int=40,
    num_attempts:int=100,
    max_iter=int(1e4),
    sigma:float=0.05
):

   # Similar to previous method:
    gkp_simmilarity_func = get_gkp_cost_function(num_moments)
    initial_state = Fock.ground_state_density_matrix(num_moments=num_moments)
    
    # Define operations:
    coherent_control = CoherentControl(num_moments=num_moments)    
    standard_operations : CoherentControl.StandardOperations  = coherent_control.standard_operations(num_intermediate_states=0)
    operations = [
        standard_operations.power_pulse_on_specific_directions(power=1, indices=[0,1,2]),
        standard_operations.power_pulse_on_specific_directions(power=2, indices=[0]),
        standard_operations.power_pulse_on_specific_directions(power=1, indices=[2]),
        standard_operations.power_pulse_on_specific_directions(power=1, indices=[0]),
        standard_operations.power_pulse_on_specific_directions(power=2, indices=[2]),
        standard_operations.power_pulse_on_specific_directions(power=1, indices=[0]),
        standard_operations.power_pulse_on_specific_directions(power=2, indices=[2]),
        standard_operations.power_pulse_on_specific_directions(power=1, indices=[0,1,2])
    ]

    ## Params:
    N = num_moments
    r = 1.1513
    theta = 0
    R = np.sqrt( (np.cosh(4*r)) / (2 * (N**2) ) )
    phi = np.arcsin( (-2*N*R) / np.sqrt(4 * (N**4) * (R**4) - 1) )
    x2 = 7.4
    z2 = pi/2
    eps = 0.1  
    _bounds = lambda: (-pi-eps, pi+eps)

    initial_values = [0.0, 0.0, 0.0, R, -phi, x2, z2, x2, z2, 0.0, 0.0, 0.0]
    num_params = len(initial_values)

    variable_boundaries = np.array([_bounds() for val in initial_values])

    @decorators.sparse_execution(50, default_results=None)
    def print_cost(cost):
        print(cost)

    def total_cost_function(theta:np.ndarray) -> float : 
        final_state = coherent_control.custom_sequence(initial_state, theta=theta, operations=operations )
        cost = gkp_simmilarity_func(final_state)
        print_cost(cost)
        return cost

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

    print(model)
    print("Done.")

    


if __name__ == "__main__":
    # _example_gkp2()
    # result = learning()
    result = learning_by_genetics()
    print("Finished main.")

