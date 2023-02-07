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
    sigma:float=0.000005
):

    # Similar to previous method:
    gkp_simmilarity_func = get_gkp_cost_function(num_moments, form="hex")
    initial_state = Fock.ground_state_density_matrix(num_moments=num_moments)
    
    # Define operations:
    coherent_control = CoherentControl(num_moments=num_moments)    
    standard_operations : CoherentControl.StandardOperations  = coherent_control.standard_operations(num_intermediate_states=0)
    operations = [
        standard_operations.power_pulse_on_specific_directions(power=1, indices=[0,1,2]),
        standard_operations.power_pulse_on_specific_directions(power=2, indices=[0]),
        standard_operations.power_pulse_on_specific_directions(power=1, indices=[2]),
        standard_operations.power_pulse_on_specific_directions(power=1, indices=[0,1,2]),
        standard_operations.power_pulse_on_specific_directions(power=2, indices=[0]),
        standard_operations.power_pulse_on_specific_directions(power=1, indices=[2]),
        standard_operations.power_pulse_on_specific_directions(power=1, indices=[0,1,2]),
        standard_operations.power_pulse_on_specific_directions(power=1, indices=[0]),
        standard_operations.power_pulse_on_specific_directions(power=2, indices=[2]),
        standard_operations.power_pulse_on_specific_directions(power=1, indices=[0,1,2]),
        standard_operations.power_pulse_on_specific_directions(power=1, indices=[0]),
        standard_operations.power_pulse_on_specific_directions(power=2, indices=[2]),
        standard_operations.power_pulse_on_specific_directions(power=1, indices=[0,1,2]),
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
    initial_value = [ 
        +0.0226849134384629 , +0.4786281740461114 , +0.0729784739310401 , +0.1140694340969037 , +0.0341110364259584 ,
        +2.0680971611351708 , +0.7002372868650459 , -2.6157394967856362 , +3.1884836600695667 , -0.2611978870511275 ,
        +0.7972339052136699 , -0.2839138306588183 , +0.0283770793970706 , +0.3349324538704883 , +0.2466455989229878 ,
        +0.3560598423687193 , +0.3684436540676541 , -0.6613483944452156 , +3.0103096677141616 , +0.7676036596645468 ,
        -0.0355497247756740 , +0.3898080459281389 , +1.2273930001579925 , +0.2167926356756356 , +1.0505091477106965 ,
        +3.2414602557490477 , +1.6329652666266483 , +0.1556180267577766
    ]


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

