# ==================================================================================== #
# |                                   Imports                                        | #
# ==================================================================================== #
if __name__ == "__main__":
    import pathlib, sys
    sys.path.append(
        str(pathlib.Path(__file__).parent.parent.parent)
    )

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
from physics.famous_density_matrices import ground_state
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
    num_atoms:int, 
    /,*,
    num_intermediate_states:int=0
) -> tuple[
    list[BaseParamType],
    list[Operation]
]:

    # Define operations:
    coherent_control = CoherentControl(num_atoms=num_atoms)    
    standard_operations : CoherentControl.StandardOperations  = coherent_control.standard_operations(num_intermediate_states=num_intermediate_states)
    rotation_op = standard_operations.power_pulse_on_specific_directions(power=1, indices=[0,1,2])
    x_op = lambda p: standard_operations.power_pulse_on_specific_directions(power=p, indices=[0])
    y_op = lambda p: standard_operations.power_pulse_on_specific_directions(power=p, indices=[1])

    # theta = [
    #     +0.5352819887733233 , -0.1217385211787736 , -0.0142376006460310 , -0.0294579569374435 , +0.0033285013050357 ,
    #     +0.1057100917549561 , +2.2125653397086520 , +1.6649087997215464 , +0.6734379053703872 , -0.0036149100923742 ,
    #     -2.8729626056070394 , +0.0069612216549833 , +0.1615236746020045 , -0.4383205554421037 , +0.2151793599692808 ,
    #     +0.3539346809289186 , -0.0435785211138870 , -0.1596301216025572 , -0.0074264171987785 , +0.0290911287827127 ,
    #     +0.3022864353719567 , +0.0424915681150559 , -0.1938454500962793 , +0.0096249649944047 , +0.0263377109894004 ,
    #     +2.3014368599893773 , -1.1756129037458227 , -3.1354002182312604 , +3.1471422782388867 , -0.0054368051299319 ,
    #     +0.2205282975754199 , +0.0552127480267320 , -0.3219987980748296 , -1.3802174649492587 , +0.0022425940708251 ,
    #     +0.7108404131301234 , +0.1556429809754983 , +0.3024989188386167 , +0.4537712232281242 , +0.2445828393778178 ,
    #     +0.4802297135680608 , +0.2146000928161127 , -0.2485823065001693 , +3.0271183941046189 , +0.7533043043314178 ,
    #     -0.0471789994017479 , +0.4899964410014738 , +1.3155765488379529 , +0.2195167773706952 , +1.0508771276463658 ,
    #     +3.1459893604827265 , +1.8050147522110631 , +0.0152073383839481
    # ]
    # theta = [
    #     +0.8268066159532392 , -0.1274134353637277 , +0.0021836312711498 , -0.0483397743101697 , +0.0139407489402983 ,
    #     +0.0002308804339485 , +1.7518371078194983 , +2.0159141536633207 , +0.6796754998894192 , -0.0036067135247529 ,
    #     -2.8697653031321106 , -0.0455089465935900 , +0.1984321615459579 , -0.4216009886537395 , +0.2162200111542030 ,
    #     +0.5378685287175453 , +0.1455091337369707 , -0.1887169337147727 , -0.0056686714453631 , +0.0309343291064059 ,
    #     +0.3594281539103044 , +0.0002957196186175 , -0.4778286714520243 , +0.0102262130984527 , +0.0160910346930154 ,
    #     +2.2719559503034947 , -1.0529308955893071 , -2.9933971212044801 , +3.1515926535897929 , -0.0068527168342284 ,
    #     +0.3465308464931329 , -0.0685618577097363 , -0.4864024894803547 , -1.3742098071930435 , +0.0070833051978510 ,
    #     +0.5813479441692364 , +0.1262603227969001 , +0.3531727945111836 , +0.4628501008475980 , +0.2462419003711060 ,
    #     +0.4894281322061852 , +0.1817539760631636 , -0.2352881373604698 , +3.0233084215949511 , +0.7484300500471042 ,
    #     -0.0535728322468648 , +0.5273360526512314 , +1.3094208850617401 , +0.2204130423801768 , +1.0508576056877841 ,
    #     +3.1515926535897929 , +1.7025254190940307 , +0.0281198495566316
    # ]
    theta = [
        +1.9113824636376942 , -1.5843527730901712 , +0.0054883997920287 , -0.1432477504032325 , +0.0012536098930238 , 
        +0.0077568551991905 , +2.3907100578553568 , +2.3009523627939092 , +0.6965460425125715 , -0.0058706596182581 , 
        -3.1492399633568127 , -0.0301250613721624 , +0.2201625612921107 , -0.4382629187742479 , +0.2174340887982453 , 
        +0.6105681788958264 , +0.5416469879024951 , -0.2557742597955543 , -0.0047832542905021 , +0.0462037373008391 , 
        +0.1779096844708992 , +0.0009426149867051 , -0.7906461562506624 , +0.0060673227571567 , +0.0144500667346377 , 
        +2.3037294715863759 , -0.6408207553948011 , -2.5788938332421507 , +3.1515926535897929 , -0.0097638011950529 , 
        +0.4342785409365232 , -0.6502104815327515 , -0.3281970185358662 , -1.3801958800870593 , +0.0056862262421313 , 
        +0.6221531410074927 , +0.1541377458376835 , +0.3479945663191109 , +0.4663041857397989 , +0.2446147764763673 , 
        +0.4734006818088673 , +0.1789609441771348 , -0.2524736497185248 , +3.0240983423453591 , +0.7506785749301168 , 
        -0.0414875177017536 , +0.5288630300213444 , +1.3132768440910452 , +0.2189500338012511 , +1.0514955655221645 , 
        +3.1374175416893761 , +1.6953013233117067 , -0.0548397235393724 
    ]

    operations = [
        rotation_op,
        x_op(2), y_op(2),
        rotation_op,
        x_op(2), y_op(2),
        rotation_op,
        x_op(2), y_op(2),
        rotation_op,
        x_op(2), y_op(2),
        rotation_op,
        x_op(2), y_op(2),
        rotation_op,
        x_op(2), y_op(2),
        rotation_op,
        x_op(2), y_op(2),
        rotation_op,
        x_op(2), y_op(2),
        rotation_op,
        x_op(2), y_op(2),
        rotation_op,
        x_op(2), y_op(2),
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
    num_atoms:int=40, 
    num_total_attempts:int=2000,
    max_iter_per_attempt=3*int(1e3),
    max_error_per_attempt=1e-11,
    num_free_params=20,
    sigma=0.00002
):

    ## Define operations and cost-function:
    cost_function = get_gkp_cost_function(num_atoms, form="hex")
    initial_state = ground_state(num_atoms=num_atoms)
    params, operations = best_sequence_params(num_atoms)

    ## Learn:
    best_result = learn_custom_operation_by_partial_repetitions(
        # Mandatory Inputs:
        initial_state=initial_state,
        cost_function=cost_function,
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

