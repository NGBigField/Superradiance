
        
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
    visuals,
    saveload,
    strings,
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
    theta : np.array = None
    similarity : float = None
    state : np.matrix = None
    time : float = None


# ==================================================================================== #
# |                                  Typing hints                                    | #
# ==================================================================================== #
_MatrixType = Union[np.matrix, np.array]

# ==================================================================================== #
# |                               Declared Functions                                 | #
# ==================================================================================== #


def learn_pulse(initial_state:_MatrixType, target_state:_MatrixType, max_iter:int=1000, x:bool=True, y:bool=True, z:bool=True, save_results:bool=False) -> LearnedResults:

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

    # cost function:
    def _cost_func(theta:np.ndarray) -> float :  
        final_state = coherent_control.pulse_on_state(state=initial_state, x=theta[0])
        diff = np.linalg.norm(final_state - target_state)
        cost = diff**2
        return cost

    # Progress_bar
    prog_bar = visuals.ProgressBar(max_iter, "Minimizing: ")
    def _after_each(xk:np.ndarray) -> False:
        prog_bar.next()

    # Opt Config:
    initial_point = np.random.random((1))
    options = dict(
        maxiter = max_iter
    )      

    # Run optimization:
    start_time = time.time()
    minimum = minimize(
        _cost_func, 
        initial_point, 
        method=OPT_METHOD, 
        options=options, 
        callback=_after_each, 
    )
    finish_time = time.time()
    optimal_theta = minimum.x
    prog_bar.close()
    
    # Pack learned-results:
    learned_results = LearnedResults(
        theta = optimal_theta,
        similarity = minimum.fun,
        time = finish_time-start_time
    )

    if save_results:
        saveload.save(learned_results, "learned_results "+strings.time_stamp())


    return learned_results


def learn_specific_state(initial_state:_MatrixType, target_state:_MatrixType, max_iter:int=100, num_pulses:int=3, save_results:bool=True) -> LearnedResults:

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
    initial_point = np.random.random((num_params))
    options = dict(
        maxiter = max_iter
    )      
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
    )
    finish_time = time.time()
    optimal_theta = minimum.x
    prog_bar.close()
    
    # Pack learned-results:
    learned_results = LearnedResults(
        theta = optimal_theta,
        state = coherent_control.coherent_sequence(initial_state, optimal_theta),
        similarity = minimum.fun,
        time = finish_time-start_time
    )

    if save_results:
        saveload.save(learned_results, "learned_results "+strings.time_stamp())


    return learned_results
    


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

# ==================================================================================== #
# |                                  main tests                                      | #
# ==================================================================================== #

def _test_learn_pi_pulse(num_moments:int=4, max_iter:int=1000) -> float:
    assertions.even(num_moments)
    initial_state = Fock.create_coherent_state(num_moments=num_moments, alpha=0.0, output='density_matrix', type_='normal')    
    target_state = Fock(num_moments).to_density_matrix(num_moments=num_moments)
    results = learn_pulse(initial_state, target_state, max_iter=max_iter, x=True)
    print(results)




def main(num_moments:int=4, max_iter:int=1000, num_pulses:int=5, plot_on:bool=False, video_on:bool=True):

    assertions.even(num_moments)
    initial_state = Fock(0).to_density_matrix(num_moments=num_moments)
    target_state = Fock(num_moments//2).to_density_matrix(num_moments=num_moments)

    if plot_on:
        visuals.plot_city(initial_state)
        visuals.plot_city(target_state)

    results = learn_specific_state(initial_state, target_state, max_iter=max_iter, num_pulses=num_pulses)
    print(f"==========================")
    print(f"num_pulses = {num_pulses}")
    print(f"run_time = {timedelta(seconds=results.time)} [hh:mm:ss]")
    print(f"similarity = {results.similarity}")

          
    coherent_control = CoherentControl(num_moments=num_moments)
    final_state = coherent_control.coherent_sequence(initial_state, theta=results.theta, record_video=video_on)
    np_utils.print_mat(final_state)

    return results


if __name__ == "__main__":
    # _test_learn_pi_pulse(num_moments=4)
    main(num_pulses=15, max_iter=100000, video_on=True)
    print("Done.")