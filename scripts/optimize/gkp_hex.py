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


def best_sequence_params(
    num_atoms:int
) -> tuple[
    list[BaseParamType],
    list[Operation]
]:

    # Define operations:
    coherent_control = CoherentControl(num_atoms=num_atoms)    
    standard_operations : CoherentControl.StandardOperations  = coherent_control.standard_operations(num_intermediate_states=0)
    rotation_op = standard_operations.power_pulse_on_specific_directions(power=1, indices=[0,1,2])
    x_op = lambda p: standard_operations.power_pulse_on_specific_directions(power=p, indices=[0])
    z_op = lambda p: standard_operations.power_pulse_on_specific_directions(power=p, indices=[2])

    theta = [
        +0.5915523671911551 , -0.1217385211787736 , -0.0151770701887521 , -0.0590498737259241 , +0.0073514128772732 , 
        -0.3433216442856554 , +1.9455471838567706 , +1.5741725015968462 , +0.6559661768289278 , -0.0022903343667889 , 
        -2.8729626056070394 , +0.0211325618835901 , +0.0395243033605317 , -0.4418813001018959 , +0.2170132947238690 , 
        +0.2177205604245520 , -0.1358160208779850 , -0.1596301216025572 , -0.0074264171987785 , +0.0280249974371016 , 
        +0.2908475934617583 , +0.0424915681150559 , -0.1845857256452919 , +0.0089144571227017 , +0.0263377109894004 , 
        +2.1877029896634914 , -1.1756129037458227 , -3.1035487121131400 , +3.1495710435850768 , -0.0063420790247544 , 
        +0.2151486062223739 , +0.0920859950360315 , -0.2887925630117823 , -1.3490315370339108 , +0.0013142448661060 , 
        +0.7132389897002627 , +0.0603174640435637 , +0.3269932227949481 , +0.4428568326476897 , +0.2454231702942572 , 
        +0.4802297135680608 , +0.2553561634623757 , -0.2405018549403924 , +2.8650820105662973 , +0.7571455611085348 , 
        -0.0140650089275624 , +0.4683369594950398 , +1.1755267138204792 , +0.2157565751363839 , +1.0463166006446798 , 
        +3.1515908141329652 , +1.7875154309132479 , +0.0182805283053670 
    ]

    operations = [
        rotation_op,
        x_op(2), z_op(2),
        rotation_op,
        x_op(2), z_op(2),
        rotation_op,
        x_op(2), z_op(2),
        rotation_op,
        x_op(2), z_op(2),
        rotation_op,
        x_op(2), z_op(2),
        rotation_op,
        x_op(2), z_op(2),
        rotation_op,
        z_op(1), z_op(2),
        rotation_op,
        x_op(2), z_op(2),
        rotation_op,
        x_op(2), z_op(2),
        rotation_op,
        x_op(2), z_op(2),
        rotation_op,
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

    return params, operations


def _example_gkp2(
    num_moments = 40
):

    # Define the basics:
    ground_state = Fock.ground_state_density_matrix(num_atoms=num_moments)    
    coherent_control = CoherentControl(num_atoms=num_moments)
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
    ground_state = Fock.ground_state_density_matrix(num_atoms=num_moments)    
    coherent_control = CoherentControl(num_atoms=num_moments)

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
    

def main(
    num_moments:int=40, 
    num_total_attempts:int=2000,
    max_iter_per_attempt=5*int(1e3),
    max_error_per_attempt=1e-12,
    num_free_params=20,
    sigma=0.00002
):

    ## Define operations and cost-function:
    gkp_simmilarity_func = get_gkp_cost_function(num_moments, form="hex")
    initial_state = Fock.ground_state_density_matrix(num_atoms=num_moments)
    params, operations = best_sequence_params(num_moments)

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
        num_attempts=num_total_attempts,
        log_name="GKP-hex "+strings.time_stamp()
    )

    ## Finish:
    sounds.ascend()
    print(best_result)
    return best_result
   


if __name__ == "__main__":
    result = main()
    print("Finished main.")

