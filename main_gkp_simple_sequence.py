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
    num_attempts:int=2000,
    max_iter=10*int(1e3),
    sigma:float=0.00005
):

    # Similar to previous method:
    gkp_simmilarity_func = get_gkp_cost_function(num_moments, form="hex")
    initial_state = Fock.ground_state_density_matrix(num_moments=num_moments)
    
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

    # initial_values = [ 
    #     0.0, 0.0, 0.0, 
    #     +3.1168508767970624 , -3.2415921418965654 , -0.3856085475202880 , +0.3213609612119441 , +0.1284533628387481 ,
    #     +0.7505659619369756 , +3.0352185363411244 , +0.5774420665601613 , -0.1625743316906557
    # ]
    # initial_values = [ 
    #     +0.0000302095022077 , -0.0003069286390748 , +0.0046979570856014 , +0.0002225095352105 , +0.0024031346776784 ,
    #     +3.0560300466112977 , +0.6489147200565115 , +0.2786231808076512 , +3.2151420618345563 , -0.9813083530218557 ,
    #     +0.2778068173244064 , -0.1980791328486351 , +0.0626205910879482 , +0.0782449795840803 , +0.2480254231074470 ,
    #     +0.1978148747493451 , +0.3210525549398842 , -0.0020982940039804 , +3.2415926535897910 , +0.7893526251973880 ,
    #     +0.1392306610724132 , +0.2878367900602354 , +0.9273613029542672 , +0.0447525198423511 , +1.0542571319609275 ,
    #     +3.2415926535897910 , +1.5716700575967337 , +0.1062400685349179
    # ]
    # initial_values = [ 
    #     2.59348887e-03,  1.53356588e-02,  1.15215870e-02,  2.20305594e-02,
    #     1.22936408e-02,  2.52039509e+00,  1.41437091e+00,  4.16112649e-02,      
    #     3.19188978e+00, -7.99415490e-01,  3.94945713e-01, -1.66471569e-01,      
    #     8.06368209e-02,  7.81635147e-02,  2.48786435e-01,  2.07262144e-01,      
    #     3.31289398e-01,  6.81665046e-03,  3.24135706e+00,  7.98047789e-01,      
    #     1.20737441e-01,  2.84761554e-01,  6.21100816e-01,  5.05027259e-02,      
    #     1.05678020e+00,  3.24159265e+00,  1.56018753e+00,  1.02745115e-01,    
    # ]
    # initial_values = [ 
    #     +0.0226951010145494 , +0.4785145273090699 , +0.0731690703517433 , 
    #     +0.1140823088947117 , 
    #     0.0, 0.0, 0.0,
    #     +0.0341458684622853 ,
    #     +0.0009787335467363 , +0.0004302618254471 , -0.0057574747532975 , 
    #     0.0, 0.0,
    #     +2.0679787335467363 , +0.7004302618254471 , -2.6157574747532975 , 
    #     +3.1884913500308025 , 
    #     0.0, 0.0, 0.0,
    #     -0.2612978043257493 ,
    #     +0.7972971070145272 , -0.2839738479586658 , +0.0284382774633870 , +0.3349387526701504 , +0.2466462814056702 ,
    #     +0.3560531087531225 , +0.3684811817470544 , -0.6614111469798650 , +3.0102702642929424 , +0.7676005868481414 ,
    #     -0.0355707240115258 , +0.3898480956441166 , +1.2274842259734895 , +0.2168299177345324 , +1.0505099656456709 ,
    #     +3.2414742353802661 , +1.6329449110245915 , +0.1556660536011137
    # ]

    # initial_values = [
    #     6.82963825e-03,  5.47705595e-01,  6.15651363e-02,  1.26822130e-01, 0.0,
    #     1.08882536e-02,  4.95675592e-05,  3.80638701e-03,  2.30573748e-01,      
    #     6.04698879e-03,  2.77114871e-03,  1.66220638e-03,  2.76017120e-03,      
    #     9.16469171e-03,  2.32863823e+00,  1.35040146e-02, -3.24148049e+00,      
    #     3.18660825e+00,  1.84540362e-03,  2.95144167e-03,  1.83630645e-03,      
    #    -5.20909529e-01,  5.27098671e-01, -2.11573415e-01,  7.24909418e-02,      
    #     4.18618847e-01,  2.52524695e-01,  5.70141278e-01,  3.42530853e-01,      
    #    -2.63217899e-01,  2.83658116e+00,  7.68692325e-01, -1.36015269e-02,      
    #     3.86870224e-01,  1.29085404e+00,  2.04897665e-01,  1.05043608e+00,
    #     3.24092788e+00,  1.64011660e+00,  1.47969712e-01
    # ]
    # initial_values = [ 
    #     6.81806335e-03,  2.39189727e+00, -8.23790005e-01,  2.60556132e-01,
    #     2.22292080e-02, -4.33093374e-02,  4.48995323e-03,  1.06571607e-03,      
    #     1.97210817e-01,  6.29573126e-03,  1.09316885e-03,  2.17375791e-03,      
    #    -6.19163827e-03,  3.27860732e-02,  2.22289535e+00,  2.54973462e-05,      
    #    -3.04117734e+00,  3.19495777e+00, -9.11080025e-03,  1.18495864e-01,      
    #     1.99116887e-03, -4.41957090e-01,  6.58592171e-01, -3.19659530e-01,      
    #     2.56990573e-02,  4.24998292e-01,  2.46566732e-01,  4.76826709e-01,      
    #     2.46359421e-01, -2.35254143e-01,  2.86613267e+00,  7.58469573e-01,      
    #     1.46553991e-02,  4.43581415e-01,  1.18611023e+00,  2.16910694e-01,
    #     1.04839569e+00,  3.10288852e+00,  1.83921808e+00,  6.33151139e-02,
    # ]
    initial_values = [ 
        1.34454016e-03,  2.41805564e+00, -6.32792228e-01,  3.62933384e-01,
        2.30525268e-03,  4.61751336e-02,  1.23499417e-04, -1.00850967e-01,      
        2.06893578e-01,  2.53052138e-01, -8.75197988e-02,  5.32881489e-04,      
       -5.78930231e-03,  5.05680698e-02,  2.50287547e+00,  4.10666561e-04,      
       -3.24155022e+00,  3.15647844e+00, -2.07153214e-03,  1.28517336e-01,      
       -1.76193295e-03, -7.60747523e-01,  5.52510571e-01, -3.75337486e-02,      
        4.63638020e-02,  4.37669695e-01,  2.46126009e-01,  4.79985646e-01,
        2.53958866e-01, -2.33949559e-01,  2.86573712e+00,  7.57144620e-01,
        6.79729365e-03,  4.44434314e-01,  1.18569091e+00,  2.15532598e-01,
        1.04687606e+00,  3.13808059e+00,  1.78770221e+00,  4.51076752e-02,
    ]



    # final_state = coherent_control.custom_sequence(initial_state, initial_values, operations)
    # visuals.plot_matter_state(final_state)
    # visuals.plot_light_wigner(final_state)

    best_result = SimpleNamespace(score=np.inf)

    theta = deepcopy(initial_values)
    for i in range(num_attempts):

        param_config : List[FreeParam] = []
        for i, val in enumerate(theta):
            param = FreeParam(
                index=i, 
                initial_guess=val+np.random.normal(1)*sigma, 
                bounds=_bounds(), 
                affiliation=None
            )   # type: ignore       
            param_config.append(param)
            
    
        result = learn_custom_operation(
            num_moments=num_moments, 
            initial_state=initial_state, 
            cost_function=gkp_simmilarity_func, 
            operations=operations, 
            max_iter=max_iter, 
            parameters_config=param_config
        )

        if result.score < best_result.score:
            print("Best result!")
            best_result = result
            theta = result.operation_params
            print(f"score={best_result.score}")
            print(f"theta={theta}")
            

    sounds.ascend()
    print(best_result)
    return best_result



def learning_by_genetics(
    num_moments:int=40
):

   # Similar to previous method:
    gkp_simmilarity_func = get_gkp_cost_function(num_moments, form="hex")
    initial_state = Fock.ground_state_density_matrix(num_moments=num_moments)
    
    # Define operations:
    coherent_control = CoherentControl(num_moments=num_moments)    
    standard_operations : CoherentControl.StandardOperations  = coherent_control.standard_operations(num_intermediate_states=0)
    operations = [
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

    # initial_values = [
    #     R, -phi, 
    #     x2, z2, 
    #     x2, z2, 
    #     0.0, 0.0, 0.0
    # ]
    initial_values = [ 
        3.13226423, -2.84102447, 
        -0.37299926,  0.33655829,  
        0.1239896 , 0.7734487 ,  
        2.21233026,  2.25752907, -0.19431447
    ]

    num_params = len(initial_values)

    variable_boundaries = np.array([_bounds() for val in initial_values])

    @decorators.sparse_execution(20, default_results=None)
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
        print(model.output_dict)
    except Exception as e:
        errors.print_traceback(e)

    print(model)
    print("Done.")

    


if __name__ == "__main__":
    # _example_gkp2()
    result = learning()
    # result = learning_by_genetics()
    print("Finished main.")

