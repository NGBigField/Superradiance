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
    coherent_control = CoherentControl(num_atoms=num_moments)    
    standard_operations : CoherentControl.StandardOperations  = coherent_control.standard_operations(num_intermediate_states=0)
    rotation_op = standard_operations.power_pulse_on_specific_directions(power=1, indices=[0,1,2])
    x_op = lambda p: standard_operations.power_pulse_on_specific_directions(power=p, indices=[0])
    z_op = lambda p: standard_operations.power_pulse_on_specific_directions(power=p, indices=[2])

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
    # theta = [
    #     0.0, 0.0, 0.0, 0.0, 0.0, 
    #     +0.1655032014411755 , +2.4832086654828904 , -0.2573326941208374 , +0.4163853941394122 , 0.0, +0.0029990870467504 , 
    #     +0.0251293663835946 , -0.0319105584715656 , -0.1423243044117304 , +0.2040111786021201 , +0.2360654828024510 , 
    #     -0.0896731445126770 , +0.0001798263579837 , -0.0057408068256298 , +0.0522726103850305 , +2.5092237630827254 , 
    #     -0.1970766968214233 , -3.2415502200000001 , +3.1564784399999999 , +0.0334753487982894 , +0.1144346418069357 , 
    #     +0.0050323946884644 , -0.8761213179244063 , +0.5716004929188870 , -0.0274091450876742 , +0.1076883013117541 ,
    #     +0.4392768720158966 , +0.2460877270237515 , +0.4805331375256521 , +0.2528870696473670 , -0.2345249200366185 ,
    #     +2.8657904298334946 , +0.7572080522212137 , +0.0078615725543411 , +0.4455474819976945 , +1.1864005198203129 ,
    #     +0.2154250602251789 , +1.0468176187533764 , +3.1397684985823000 , +1.7862455947608149 , +0.0431204798126949 ,
    # ]
    # theta = [
    #     +0.0001450601518824 , +0.0000065770704199 , +0.0000006837869571 , -0.0000138322510140 , -0.0000048973041840 , 
    #     -0.9416240243159255 , +1.5186075892864470 , +0.5831666599402248 , +0.6776976813641020 , -0.0017529000871072 , 
    #     -1.1274992394993519 , -0.0065716276335625 , +0.0242663568490457 , -0.3829728716366175 , +0.2129366683022240 , 
    #     +0.3153030556578900 , -0.0997728568548071 , -0.0664202185466046 , -0.0048957677626110 , +0.0511740593738266 , 
    #     +2.3738805008691459 , -0.9873591842527798 , -3.1465232728948500 , +3.1515683622789754 , +0.1700795484189499 , 
    #     +0.0989695492294589 , -0.1214959930072439 , -1.0976030284772107 , +0.7484551103748885 , +0.0375025800213373 , 
    #     +0.1982634603102721 , +0.4410307564301735 , +0.2456731458222782 , +0.4827442200443236 , +0.2533425763088177 ,
    #     -0.2379135843541080 , +2.8651571458312191 , +0.7570695583196858 , +0.0061668901624979 , +0.4518601962409419 ,
    #     +1.1800823901708919 , +0.2160482029734905 , +1.0462183308229851 , +3.1463983608428689 , +1.7940268809088837 ,
    #     +0.0136859746949420
    # ]
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

    return operations, params


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
    

def learn_sx2_pulses(
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
        num_attempts=num_total_attempts,
        log_name="GKP-hex "+strings.time_stamp()
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

