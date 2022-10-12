
        
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
    Optional,
    Callable,
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

# For coherent control
from coherentcontrol import (
    S_mats,
    pulse,
    CoherentControl,
    _DensityMatrixType,
)

# for optimization:
from scipy.optimize import minimize, OptimizeResult, show_options  # for optimization:   
        
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
NUM_PULSE_PARAMS : Final = 4  
TOLERANCE = 1e-10

# ==================================================================================== #
# |                                    Classes                                       | #
# ==================================================================================== #
@dataclass
class LearnedResults():
    theta : np.array = None
    similarity : float = None
    initial_state : np.matrix = None
    final_state : np.matrix = None
    time : float = None
    iterations : int = None

    def __repr__(self) -> str:
        np_utils.fix_print_length()
        newline = '\n'
        s = ""
        s += f"similarity={self.similarity}"+newline
        s += f"theta={self.theta}"+newline
        s += f"run-time={self.time}"+newline
        s += np_utils.mat_str_with_leading_text(self.initial_state, text="initial_state: ")+newline       
        s += np_utils.mat_str_with_leading_text(self.final_state  , text="final_state  : ")+newline  
        s += f"num_iterations={self.iterations}"+newline
        return s
    


# ==================================================================================== #
# |                                  Typing hints                                    | #
# ==================================================================================== #



# ==================================================================================== #
# |                                Inner Functions                                   | #
# ==================================================================================== #

def _coherent_control_from_mat(mat:_DensityMatrixType) -> CoherentControl:
    # Set basic properties:
    matrix_size = mat.shape[0]
    max_state_num = matrix_size-1
    return CoherentControl(max_state_num)

def _deal_bounds(num_params:int) -> int: 
    def _bound_rule(i:int) -> Tuple[Any, Any]:
        # Derive:
        if i%NUM_PULSE_PARAMS==3:
            return (0, None)
        else:
            return (-np.pi, +np.pi)
    # Define bounds:
    return [_bound_rule(i) for i in range(num_params)]

def _deal_initial_guess(num_params:int, initial_guess:Optional[np.array]) -> np.array :
    time_indices = np.arange(NUM_PULSE_PARAMS-1, num_params, NUM_PULSE_PARAMS)
    if initial_guess is not None:  # If guess is given:
        assert len(initial_guess) == num_params, f"Needed number of parameters for the initial guess is {num_params}"
        if isinstance(initial_guess, list):
            initial_guess = np.array(initial_guess)
        assert np.all(initial_guess[time_indices]>=0), f"All decay-times must be non-negative!"    
    else:  # if we need to create a guess:    
        initial_guess = np.random.normal(0, np.pi/2, (num_params))
        initial_guess[time_indices] = np.abs(initial_guess[time_indices])
    return initial_guess
    

def _common_learn(
    initial_state : _DensityMatrixType, 
    cost_function : Callable[[_DensityMatrixType], float],
    max_iter : int,
    num_pulses : int,
    initial_guess : Optional[np.array] = None,
    save_results : bool=True
) -> LearnedResults:
    
    # Set basic properties:
    coherent_control = _coherent_control_from_mat(initial_state)
    num_params = CoherentControl.num_params_for_pulse_sequence(num_pulses=num_pulses)

    # Progress_bar
    prog_bar = visuals.ProgressBar(max_iter, "Minimizing: ")
    def _after_each(xk:np.ndarray) -> False:
        prog_bar.next()

    # Opt Config:
    initial_guess = _deal_initial_guess(num_params, initial_guess)
    options = dict(
        maxiter = max_iter,
        ftol=TOLERANCE,
    )      
    bounds = _deal_bounds(num_params)      

    # Run optimization:
    start_time = time.time()
    opt_res : OptimizeResult = minimize(
        cost_function, 
        initial_guess, 
        method=OPT_METHOD, 
        options=options, 
        callback=_after_each, 
        bounds=bounds    
    )
    finish_time = time.time()
    optimal_theta = opt_res.x
    prog_bar.close()
    
    # Pack learned-results:
    learned_results = LearnedResults(
        theta = optimal_theta,
        similarity = opt_res.fun,
        time = finish_time-start_time,
        initial_state = initial_state,
        final_state = coherent_control.coherent_sequence(initial_state, optimal_theta),
        iterations = opt_res.nit,
    )

    if save_results:
        saveload.save(learned_results, "learned_results "+strings.time_stamp())


    return learned_results

# ==================================================================================== #
# |                               Declared Functions                                 | #
# ==================================================================================== #

def learn_midladder_state(
    initial_state:_DensityMatrixType,  
    max_iter : int=1000, 
    num_pulses : int=5, 
    initial_guess : Optional[np.array] = None,
    save_results : bool=True,
) -> LearnedResults:

    # Check inputs:
    assertions.density_matrix(initial_state)

    # Constants:
    @dataclass
    class _Cost():
        midvalue   = -1.0
        diagonal   = +2.0
        every_else = +0.5

    # cost function:
    coherent_control = _coherent_control_from_mat(initial_state)
    num_moments = coherent_control.num_moments
    target_state = Fock(num_moments//2).to_density_matrix(num_moments)
    def _cost_func(theta:np.ndarray) -> float :  
        final_state = coherent_control.coherent_sequence(state=initial_state, theta=theta)
        total_cost = 0.0
        shape = final_state.shape
        for i in range(shape[0]):
            for j in range(shape[1]):
                val = final_state[i,j]
                if i==j: # diagonal
                    if i==shape[0]//2:
                        cost = _Cost.midvalue
                    else:
                        cost = _Cost.diagonal
                else:
                    cost = _Cost.every_else
                total_cost += cost*abs(val)
        return total_cost

    # Call base function:
    return _common_learn(
        initial_state=initial_state,
        max_iter=max_iter,
        num_pulses=num_pulses,
        cost_function=_cost_func,
        save_results=save_results,
        initial_guess=initial_guess,
    )


def learn_specific_state(
    initial_state:_DensityMatrixType, 
    target_state:_DensityMatrixType, 
    max_iter : int=100, 
    num_pulses : int=3, 
    initial_guess : Optional[np.array] = None,
    save_results : bool=True,
) -> LearnedResults:

    # Check inputs:
    assertions.density_matrix(initial_state)
    assertions.density_matrix(target_state)
    assert initial_state.shape == target_state.shape
 
    # cost function:
    coherent_control = _coherent_control_from_mat(initial_state)
    def _cost_func(theta:np.ndarray) -> float :  
        final_state = coherent_control.coherent_sequence(state=initial_state, theta=theta)
        diff = np.linalg.norm(final_state - target_state)
        cost = diff**2
        return cost
    # Call base function:
    return _common_learn(
        initial_state=initial_state,
        max_iter=max_iter,
        num_pulses=num_pulses,
        cost_function=_cost_func,
        save_results=save_results,
        initial_guess=initial_guess,
    )

    
# ==================================================================================== #
# |                                  main tests                                      | #
# ==================================================================================== #

def run_many_guesses(
    max_num_pulses:int=16, 
    num_tries:int=10,
    num_moments:int=8
) -> LearnedResults:

    # Track the best results:
    best_results : LearnedResults = LearnedResults(similarity=1e10) 

    # For movie:
    coherent_control = CoherentControl(num_moments=num_moments)
    movie_config = CoherentControl.MovieConfig(
        active=True,
        show_now=False,
        num_transition_frames=5,
        num_freeze_frames=5,
        fps=2,
        bloch_sphere_resolution=10
    )    
    
    for num_pulses in range(1, max_num_pulses+1):
        for _ in range(num_tries):
            results = run_signle_guess(num_pulses=num_pulses, num_moments=num_moments)
            if results.similarity < best_results.similarity:
                best_results = results
                # Print and record movie:
                print(results)
                coherent_control.coherent_sequence(results.initial_state, theta=results.theta, movie_config=movie_config)
                
    saveload.save(best_results, "best_results "+strings.time_stamp())
    print("\n")
    print("\n")
    print("best_results:")
    print(best_results)
    return best_results


def run_signle_guess(
    num_moments:int=8, 
    max_iter:int=1000, 
    num_pulses:int=5, 
) -> LearnedResults:

    ## Learning Inputs:
    ####################
    assertions.even(num_moments)
    initial_state = Fock( num_moments  ).to_density_matrix(num_moments=num_moments)
    # target_state  = Fock(num_moments//2).to_density_matrix(num_moments=num_moments)
    # initial_guess = None  # [0]*CoherentControl.num_params_for_pulse_sequence(num_pulses) 
    if False:
        visuals.plot_city(initial_state)
        visuals.plot_city(target_state)

    ## STUDY:
    ##########
    # results = learn_specific_state(initial_state, target_state, max_iter=max_iter, num_pulses=num_pulses, initial_guess=initial_guess)
    results = learn_midladder_state(initial_state=initial_state, max_iter=max_iter, num_pulses=num_pulses)
    

    return results


if __name__ == "__main__":
    # _test_learn_pi_pulse(num_moments=4)
    # show_options(method=OPT_METHOD)
    results = run_many_guesses()
    print("Done.")