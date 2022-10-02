
        
# ==================================================================================== #
# |                                   Imports                                        | #
# ==================================================================================== #

# Everyone needs numpy:
import numpy as np

# For typing hints:
from typing import (
    Any,
    Tuple,
    List,
    Union,
    Dict,
    Final,
)

# import our helper modules
from utils import (
    assertions,
    numpy_tools as np_utils,
    visuals
)

# For defining coherent states:
from quantum_states.fock import Fock

# For states
from evolution import init_state, Params, CommonStates    

# For coherent control
from coherentcontrol import (
    S_mats,
    pulse,
    CoherentControl,
)

# for optimization:
from scipy.optimize import minimize, OptimizeResult  # for optimization:   
        
# For measuring time:
import time
from datetime import timedelta

# For visualizations:
import matplotlib.pyplot as plt  # for plotting test results:
from light_wigner.main import visualize_light_from_atomic_density_matrix

# For OOP:
from dataclasses import dataclass

# ==================================================================================== #
# |                                  Constants                                       | #
# ==================================================================================== #
OPT_METHOD : Final = 'SLSQP'

# ==================================================================================== #
# |                                    Classes                                       | #
# ==================================================================================== #
@dataclass
class LearnedResults():
    theta : np.array
    similarity : float
    state : np.matrix
    time : float


# ==================================================================================== #
# |                                  Typing hints                                    | #
# ==================================================================================== #
_MatrixType = Union[np.matrix, np.array]

# ==================================================================================== #
# |                               Declared Functions                                 | #
# ==================================================================================== #


def learn_specific_state(initial_state:_MatrixType, target_state:_MatrixType, max_iter:int=100, num_pulses:int=3 ) -> LearnedResults:

    # Check inputs:
    for state in [initial_state, target_state]:
        assertions.density_matrix(state)
        assert len(state.shape)==2
        assert state.shape[0] == state.shape[1]
    assert initial_state.shape[0] == target_state.shape[0]
    
    # Set basic properties:
    matrix_size = state.shape[0]
    max_state_num = matrix_size-1
    coherent_control = CoherentControl(max_state_num)
    num_params = CoherentControl.num_params_for_pulse_sequence(num_pulses=num_pulses)

    # cost function:
    def _cost_func(theta:np.ndarray) -> float :  
        final_state = coherent_control.coherent_sequence(state=initial_state, theta=theta)
        diff = np.linalg.norm(final_state - target_state)
        cost = diff**2
        return cost

    # Progress_bar
    prog_bar = visuals.ProgressBar(max_iter, "Minimizing: ")
    def _after_each(xk:np.ndarray) -> False:
        prog_bar.next()

    # Opt Config:
    initial_point = np.array([0.0]*num_params)
    options = dict(
        maxiter = max_iter
    )      
    constraints = _deal_constraints(num_params)      
    bounds = _deal_bounds(num_params)      

    # Run optimization:
    start_time = time.time()
    minimum = minimize(
        _cost_func, 
        initial_point, 
        method=OPT_METHOD, 
        options=options, 
        callback=_after_each, 
        bounds=bounds
        # constraints=constraints
    )
    finish_time = time.time()
    optimal_theta = minimum.x
    prog_bar.close()
    
    # Pack learned-results:
    return LearnedResults(
        theta = optimal_theta,
        state = coherent_control.coherent_sequence(initial_state, optimal_theta),
        similarity = minimum.fun,
        time = finish_time-start_time
    )
    


# ==================================================================================== #
# |                                Inner Functions                                   | #
# ==================================================================================== #

def _inequality_func(theta:np.ndarray, *args )->float:
    # Parse inputs:
    index = args[0]
    # Generate value ( should be positive on scipy's check, for it to be legit )
    time = theta[index]
    if time<0:
        print(time)
        return abs(time)
    return time

def _bound_rule(i:int) -> Tuple[Any, Any]:
    # Constant:
    NUM_PULSE_PARAMS : Final = 4  
    # Derive:
    if i%NUM_PULSE_PARAMS==3:
        return (0, None)
    else:
        return (None, None)

def _deal_bounds(num_params:int) -> int: 
    # Define bounds:
    bounds = [_bound_rule(i) for i in range(num_params)]
    return bounds

def _deal_constraints(num_params:int) -> int:    
    # Constants:
    NUM_PULSE_PARAMS : Final = 4
    # Init list:
    constraints : List[dict] = []
    # Iterate:
    for i in range(3, num_params, NUM_PULSE_PARAMS):
        print(i)

        constraints.append(
            dict(
                type = 'ineq',
                fun  = _inequality_func,
                args = [i]
            )
        )

    return constraints


def _learn_pi_pulse(num_iter:int=4, N:int=2, plot_on:bool=False) -> LearnedResults :
    coherent_control = CoherentControl(N)

    # init:
    params = Params(N=N)
    rho_initial = init_state(params, CommonStates.Ground)
    rho_target  = init_state(params, CommonStates.FullyExcited)

    # Helper functions:
    def _apply_pulse_on_initial_state(theta:np.array) -> np.matrix: 
        x = theta[0]
        y = theta[1]
        z = theta[2]
        return coherent_control.pulse_on_state(rho_initial, x, y, z)

    def _derive_cost_function(theta:np.array) -> float :  
        rho_final = _apply_pulse_on_initial_state(theta)
        diff = np.linalg.norm(rho_final-rho_target)
        cost = diff**2
        return cost

    def _find_optimum()->OptimizeResult:
        initial_point = np.array([0.0]*3)
        options = dict(
            maxiter = num_iter
        )            
        # Run optimization:
        start_time = time.time()
        minimum = minimize(_derive_cost_function, initial_point, method=OPT_METHOD, options=options)
        finish_time = time.time()

        # Unpack results:
        run_time = finish_time-start_time
        print(f"run_time={run_time} [sec]")
        return minimum

    # Minimize:
    opt = _find_optimum()

    # Unpack results:    
    theta = opt.x
    assert len(theta)==3

    rho_final = _apply_pulse_on_initial_state(theta)
    np_utils.print_mat(rho_final)

    # visualizing light:
    if plot_on:
        title = f"MAX_ITER={num_iter} \n{theta} "
        plot_city(rho_final, title=title)

    # Pack results:
    res = LearnedResults(theta=theta, state=rho_final)
    return res




# ==================================================================================== #
# |                                  main tests                                      | #
# ==================================================================================== #


def _test_learn_pi_pulse():
    for num_iter in [1, 2, 5, 10, 20]:
        res = _learn_pi_pulse(num_iter=num_iter, plot_on=True)
        visuals.save_figure(file_name=f"learn_pi_pulse num_iter {num_iter}")

def _test_learn_state(num_moments:int=4, max_iter:int=100, num_pulses:int=1, plot_on:bool=False):

    assertions.even(num_moments)
    initial_state = Fock.create_coherent_state(num_moments=num_moments, alpha=0.0, output='density_matrix', type_='normal')
    target_state = Fock.create_coherent_state(num_moments=num_moments, alpha=1.0, output='density_matrix', type_='normal')

    if plot_on:
        visuals.plot_city(initial_state)
        visuals.plot_city(target_state)

    # np_utils.print_mat(rho_initial)
    # np_utils.print_mat(rho_target)

    results = learn_specific_state(initial_state, target_state, max_iter=max_iter, num_pulses=num_pulses)
    print(f"==========================")
    print(f"num_pulses = {num_pulses}")
    print(f"run_time = {timedelta(seconds=results.time)} [hh:mm:ss]")
    print(f"similarity = {results.similarity}")
    # print(f"theta = {results.theta}")

    if plot_on:
        visuals.plot_city(results.state)
        visualize_light_from_atomic_density_matrix(results.state, num_moments)

    return results


if __name__ == "__main__":
    # _test_learn_pi_pulse_only_x()
    # _test_learn_pi_pulse()
    _test_learn_state(num_pulses=3, max_iter=1000)
    print("Done.")