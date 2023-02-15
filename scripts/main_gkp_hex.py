# ==================================================================================== #
# |                                   Imports                                        | #
# ==================================================================================== #
if __name__ == "__main__":
    import pathlib, sys
    sys.path.append(str(pathlib.Path(__file__).parent.parent))

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

# For defining coherent states and gkp states:
from physics.fock import Fock
from physics.gkp import get_gkp_cost_function

# For coherent control
from algo.coherentcontrol import CoherentControl, _DensityMatrixType, Operation

# For optimizations:
from algo.optimization import LearnedResults, learn_custom_operation_by_partial_repetitions, FreeParam, BaseParamType, learn_single_op


# ==================================================================================== #
#|                                helper functions                                    |#
# ==================================================================================== #

def _get_final_state(
    ground_state:_DensityMatrixType, coherent_control:CoherentControl,
    x1, x2
):
    # Basic pulses:
    Sy = coherent_control.s_pulses.Sy
    Sy2 : np.matrix = Sy@Sy  # type: ignore    
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


def _best_operations_and_values_so_far(
    num_moments:int
) -> tuple[
    list[Operation],
    list[BaseParamType]
]:

    # Define operations:
    coherent_control = CoherentControl(num_moments=num_moments)    
    standard_operations : CoherentControl.StandardOperations  = coherent_control.standard_operations(num_intermediate_states=0)
    rotation_op = standard_operations.power_pulse_on_specific_directions(power=1, indices=[0,1,2])
    x_op = lambda p: standard_operations.power_pulse_on_specific_directions(power=p, indices=[0])
    z_op = lambda p: standard_operations.power_pulse_on_specific_directions(power=p, indices=[2])

    operations = [
        rotation_op,
        x_op(2),
        rotation_op,
        x_op(2), z_op(2),
        rotation_op,
        x_op(2), z_op(2),
        rotation_op,
        x_op(2),
        rotation_op,
        z_op(1),
        rotation_op,
        x_op(2), z_op(2),
        rotation_op,
        x_op(2), z_op(2),
        rotation_op,
        x_op(2), z_op(2),
        rotation_op,
    ]

    # theta = [ 
    #     1.34454016e-03,  2.41805564e+00, -6.32792228e-01,  3.62933384e-01,
    #     2.30525268e-03,  4.61751336e-02,  1.23499417e-04, -1.00850967e-01,      
    #     2.06893578e-01,  2.53052138e-01, -8.75197988e-02,  5.32881489e-04,      
    #    -5.78930231e-03,  5.05680698e-02,  2.50287547e+00,  4.10666561e-04,      
    #    -3.24155022e+00,  3.15647844e+00, -2.07153214e-03,  1.28517336e-01,      
    #    -1.76193295e-03, -7.60747523e-01,  5.52510571e-01, -3.75337486e-02,      
    #     4.63638020e-02,  4.37669695e-01,  2.46126009e-01,  4.79985646e-01,
    #     2.53958866e-01, -2.33949559e-01,  2.86573712e+00,  7.57144620e-01,
    #     6.79729365e-03,  4.44434314e-01,  1.18569091e+00,  2.15532598e-01,
    #     1.04687606e+00,  3.13808059e+00,  1.78770221e+00,  4.51076752e-02,
    # ]
    theta = [
        +0.0014395978658212 , +2.4179009713024868 , -0.6328352168485074 , +0.3627197468432675 , +0.0024305234686808 , 
        +0.0463644321744657 , +0.0002964903867057 , -0.1006371519979180 , +0.2068562398822839 , +0.2530901701906671 , 
        -0.0875367661046861 , +0.0021080127671757 , -0.0057951775870336 , +0.0506029036376613 , +2.5048685704248781 , 
        +0.0002630534477134 , -3.2415502200000001 , +3.1564784399999999 , -0.0007967779120016 , +0.1287193670720429 , 
        -0.0010812204768175 , -0.7611430776253378 , +0.5521607841235421 , -0.0369091211000953 , +0.0457234207375566 ,
        +0.4376612316902114 , +0.2461307392310771 , +0.4799618437760237 , +0.2539856443947800 , -0.2339513622891821 ,
        +2.8657363872415047 , +0.7571422233353410 , +0.0068696711042072 , +0.4443749014893962 , +1.1857226280051387 ,
        +0.2155460149460754 , +1.0468792751792380 , +3.1381748207756250 , +1.7876802380733787 , +0.0452933922151622 ,
    ]

    eps = 0.01
    params = []
    for i, val in enumerate(theta):
        param = FreeParam(
            index=i, 
            initial_guess=val, 
            bounds=(-pi-eps, pi+eps), 
            affiliation=None
        )   # type: ignore       
        params.append(param)

    return operations, params


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
    

def learn_sx2_pulses(
    num_moments:int=40, 
    num_total_attempts:int=2000,
    max_iter_per_attempt=5*int(1e3),
    max_error_per_attempt=1e-9,
    num_free_params=25,
    sigma=0.0000002
):

    ## Define operations and cost-function:
    gkp_simmilarity_func = get_gkp_cost_function(num_moments, form="hex")
    initial_state = Fock.ground_state_density_matrix(num_moments=num_moments)
    operations, params = _best_operations_and_values_so_far(num_moments)

    ## Learn:
    best_result = learn_custom_operation_by_partial_repetitions(
        # Mandatory Inputs:
        initial_state=initial_state,
        cost_function=gkp_simmilarity_func,
        operations=operations,
        initial_params=params,
        # Huristic Params:
        max_iter_per_attempt=max_iter_per_attempt,
        max_error_per_attempt=max_error_per_attempt,
        num_free_params=num_free_params,
        sigma=sigma,
        num_attempts=num_total_attempts
    )

    ## Finish:
    sounds.ascend()
    print(best_result)
    return best_result
   


if __name__ == "__main__":
    # _example_gkp2()
    result = learn_sx2_pulses()
    # result = learning_by_genetics()
    print("Finished main.")

