
        
# ==================================================================================== #
# |                                   Imports                                        | #
# ==================================================================================== #

# Everyone needs numpy:
import numpy as np
from numpy import pi

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
    errors,
    args,
)

# For defining coherent states:
from fock import Fock

# For coherent control
from coherentcontrol import (
    CoherentControl,
    _DensityMatrixType,
    Operation,
)

# for optimization:
from scipy.optimize import minimize, OptimizeResult, show_options  # for optimization:   
import metrics 
import gkp 
        
# For measuring time:
import time

# For OOP:
from dataclasses import dataclass
from enum import Enum, auto

# for plotting stuff wigner
import qutip
import matplotlib.pyplot as plt

# ==================================================================================== #
# |                                  Constants                                       | #
# ==================================================================================== #
OPT_METHOD : Final = "Nelder-Mead" #'SLSQP' # 'Nelder-Mead'
NUM_PULSE_PARAMS : Final = 4  
TOLERANCE = 1e-32

T4_PARAM_INDEX = 5

# ==================================================================================== #
# |                                    Classes                                       | #
# ==================================================================================== #
@dataclass
class LearnedResults():
    theta : np.array = None
    score : float = None
    initial_state : np.matrix = None
    final_state : np.matrix = None
    time : float = None
    iterations : int = None
    operations : List[Operation] = None

    def __repr__(self) -> str:
        np_utils.fix_print_length()
        newline = '\n'
        s = ""
        s += f"score={self.score}"+newline
        s += f"theta={self.theta}"+newline
        s += f"run-time={self.time}"+newline
        s += np_utils.mat_str_with_leading_text(self.initial_state, text="initial_state: ")+newline       
        s += np_utils.mat_str_with_leading_text(self.final_state  , text="final_state  : ")+newline  
        s += f"num_iterations={self.iterations}"+newline
        s += f"operations: {self.operations}"+newline
        return s
    
class Metric(Enum):
    NEGATIVITY = auto
    PURITY = auto
    

# ==================================================================================== #
# |                                  Typing hints                                    | #
# ==================================================================================== #



# ==================================================================================== #
# |                                Inner Functions                                   | #
# ==================================================================================== #

def _wigner(state:_DensityMatrixType, title:Optional[str]=None)->None:
    fig, ax = qutip.plot_wigner( qutip.Qobj(state) )
    if title is not None:
        ax.set_title(title)

def _initial_guess() -> List[float] :
    omega = 0.2 * 2 * np.pi
    t_1 = 2.074 * omega
    t_2 = 0.285 * omega
    t_3 = 0.191 * omega
    t_4 = 2.084 * omega
    delta_1 =  -4.0 * 2 * np.pi / omega 
    delta_2 = -18.4 * 2 * np.pi / omega
    phi_3 = 0.503
    phi_4 = 0.257

    # return [  t_1,     t_2,     delta_1,     t_3,       phi_3,    t_4,        phi_4,    delta_2 ]
    return [ 2.5066,    0.238 ,  -32.9246,    0.58  ,    0.5366,    2.1576,    0.1602, -107.5689]
    # return [  2.5066,    0.238 ,  -32.9246,    0.58  ,    0.5366,    2.1576/4,    0.1602, -107.5689]



def _coherent_control_from_mat(mat:_DensityMatrixType) -> CoherentControl:
    # Set basic properties:
    matrix_size = mat.shape[0]
    max_state_num = matrix_size-1
    return CoherentControl(max_state_num)

def _deal_bounds(num_params:int, positive_indices:np.ndarray, rotation_indices:List[int]=[]) -> int: 
    def _bound_rule(i:int) -> Tuple[Any, Any]:
        # Derive:
        if i in positive_indices:
            return (0, None)
        elif i in rotation_indices:
            return (-np.pi, +np.pi)
        else:
            return (None, None)

    # Define bounds:
    return [_bound_rule(i) for i in range(num_params)]

def _positive_indices_from_operations(operations:List[Operation]) -> np.ndarray:
    low = 0
    positive_indices : List[int] = []
    for op in operations:
        high = low + op.num_params 
        if op.positive_params_only:
            indices = list( range(low, high) )
            positive_indices.extend(indices)
        low = high
    return positive_indices

def _deal_initial_guess(num_params:int, initial_guess:Optional[np.array]) -> np.ndarray :
    positive_indices = np.arange(NUM_PULSE_PARAMS-1, num_params, NUM_PULSE_PARAMS)
    return _deal_initial_guess_common(num_params=num_params, initial_guess=initial_guess, positive_indices=positive_indices)
    
def _deal_initial_guess_common(num_params:int, initial_guess:Optional[np.array], positive_indices:np.ndarray) -> np.ndarray:
    if initial_guess is not None:  # If guess is given:
        assert len(initial_guess) == num_params, f"Needed number of parameters for the initial guess is {num_params}"
        if isinstance(initial_guess, list):
            initial_guess = np.array(initial_guess)
        if len(positive_indices)>0:
            assert np.all(initial_guess[positive_indices]>=0), f"All decay-times must be non-negative!"    
    else:  # if we need to create a guess:    
        initial_guess = np.random.normal(0, np.pi/2, (num_params))
        if len(positive_indices)>0:
            initial_guess[positive_indices] = np.abs(initial_guess[positive_indices])
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
    def _after_each(xk:np.ndarray) -> bool:
        prog_bar.next()
        finish : bool = False
        return finish

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
        score = opt_res.fun,
        time = finish_time-start_time,
        initial_state = initial_state,
        final_state = coherent_control.coherent_sequence(initial_state, optimal_theta),
        iterations = opt_res.nit,
    )

    if save_results:
        saveload.save(learned_results, "learned_results "+strings.time_stamp())


    return learned_results

def _score_str_func(state:_DensityMatrixType) -> str:
    s =  f"Purity     = {purity(state)} \n"
    s += f"Negativity = {negativity(state)}"
    # s += f"Fidelity = {fidelity(crnt_state, target_state)} \n"
    # s += f"Distance = {distance(crnt_state, target_state)} "
    return s



# ==================================================================================== #
# |                               Declared Functions                                 | #
# ==================================================================================== #

def learn_optimized_metric(
    initial_state : _DensityMatrixType,
    metric : Metric = Metric.NEGATIVITY,
    max_iter : int=1000, 
    num_pulses : int=5, 
    initial_guess : Optional[np.array] = None,
    save_results : bool=True,
) -> LearnedResults:
    
    # Choose cost-function:
    if metric is Metric.PURITY:
        measure = lambda state: purity(state)
    elif metric is Metric.NEGATIVITY:
        measure = lambda state: (-1)*negativity(state)
    else:
        raise ValueError("Not a valid option")
    
    coherent_control = _coherent_control_from_mat(initial_state)
    def _cost_func(theta) -> float:
        final_state = coherent_control.coherent_sequence(state=initial_state, theta=theta)        
        return measure(final_state)        

    # Call base function:
    return _common_learn(
        initial_state=initial_state,
        max_iter=max_iter,
        num_pulses=num_pulses,
        cost_function=_cost_func,
        save_results=save_results,
        initial_guess=initial_guess,
    )
    
    

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
        diagonal   = +5.0
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
        cost = fidelity(initial_state, final_state) * -1
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

def learn_custom_operation(    
    num_moments : int,
    initial_state : _DensityMatrixType,
    operations : List[Operation],
    cost_function : Callable[[_DensityMatrixType], float],
    max_iter : int=100, 
    initial_guess : Optional[np.array] = None,
    save_results : bool=True,
) -> LearnedResults:

    # Progress_bar
    prog_bar = visuals.ProgressBar(max_iter, "Minimizing: ")
    def _after_each(xk:np.ndarray) -> bool:
        prog_bar.next()
        finish : bool = False
        return finish

    # Opt Config:
    options = dict(
        maxiter = max_iter,
        ftol=TOLERANCE,
    )      
    positive_indices = _positive_indices_from_operations(operations)
    num_params = sum([op.num_params for op in operations])
    initial_guess = _deal_initial_guess_common(num_params=num_params, initial_guess=initial_guess, positive_indices=positive_indices)
    bounds = _deal_bounds(num_params, positive_indices)  

    
    coherent_control = CoherentControl(num_moments)

    # matrix_size = initial_state.shape[0]
    # target_state = np.zeros(shape=(matrix_size,matrix_size))
    # for i in [0, matrix_size-1]:
    #     for j in [0, matrix_size-1]:
    #         target_state[i,j]=0.5
    def total_cost_function(theta:np.ndarray) -> float : 
        final_state = coherent_control.custom_sequence(initial_state, theta=theta, operations=operations )
        cost = cost_function(final_state)
        return cost

    # Run optimization:
    start_time = time.time()
    opt_res : OptimizeResult = minimize(
        total_cost_function, 
        initial_guess, 
        method=OPT_METHOD, 
        options=options, 
        callback=_after_each, 
        bounds=bounds    
    )
    finish_time = time.time()
    prog_bar.close()
    
    # Pack learned-results:
    optimal_theta = opt_res.x
    final_state = coherent_control.custom_sequence(initial_state, theta=optimal_theta, operations=operations )
    learned_results = LearnedResults(
        theta = optimal_theta,
        score = opt_res.fun,
        time = finish_time-start_time,
        initial_state = initial_state,
        final_state = final_state,
        iterations = opt_res.nit
    )

    if save_results:
        saveload.save(learned_results, "learned_results "+strings.time_stamp())


    return learned_results    
    
# ==================================================================================== #
# |                                  main tests                                      | #
# ==================================================================================== #

def _run_many_guesses(
    min_num_pulses:int=3,
    max_num_pulses:int=16, 
    num_tries:int=5,
    num_moments:int=8
) -> LearnedResults:

    # Track the best results:
    best_results : LearnedResults = LearnedResults(score=1e10) 

    # For movie:
    # target_state  = Fock(num_moments//2).to_density_matrix(num_moments=num_moments)
    coherent_control = CoherentControl(num_moments=num_moments)
    movie_config = CoherentControl.MovieConfig(
        active=True,
        show_now=False,
        num_transition_frames=10,
        num_freeze_frames=5,
        fps=3,
        bloch_sphere_resolution=25,
        score_str_func = lambda state: _score_str_func(state)
    )    
    
    
    for num_pulses in range(min_num_pulses, max_num_pulses+1):
        for _ in range(num_tries):
            # Run:
            try:
                results = creating_gkp_algo(num_pulses=num_pulses, num_moments=num_moments)
            except Exception as e:
                errors.print_traceback(e)
            # Check if better than best:
            if results.score < best_results.score:
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



def creating_gkp_algo(
    num_moments:int=40, 
    max_iter:int=10000, 
) -> LearnedResults:

    ## Check inputs:
    assertions.even(num_moments)

    ## Define operations:
    coherent_control = CoherentControl(num_moments=num_moments)
    standard_operations : CoherentControl.StandardOperations = coherent_control.standard_operations(num_intermediate_states=0)
    Sp = coherent_control.s_pulses.Sp
    Sx = coherent_control.s_pulses.Sx
    Sy = coherent_control.s_pulses.Sy
    Sz = coherent_control.s_pulses.Sz

    ## Define initial state and guess:
    initial_state = Fock.excited_state_density_matrix(num_moments)
    initial_guess = _initial_guess()

    ## Learn how to prepare a cat state:
    noon_creation_operations = [
        standard_operations.power_pulse_on_specific_directions(power=1, indices=[0]),
        standard_operations.stark_shift_and_rot(stark_shift_indices=[1], rotation_indices=[0]),
        standard_operations.stark_shift_and_rot(stark_shift_indices=[] , rotation_indices=[0, 1]),
        standard_operations.stark_shift_and_rot(stark_shift_indices=[1], rotation_indices=[0, 1]),
    ]
    matrix_size = initial_state.shape[0]
    target_state = np.zeros(shape=(matrix_size,matrix_size))
    for i in [0, matrix_size-1]:
        for j in [0, matrix_size-1]:
            target_state[i,j]=0.5
    def cost_function(final_state:_DensityMatrixType) -> float : 
        return (-1) * metrics.fidelity(final_state, target_state)
    results = learn_custom_operation(
        num_moments=num_moments, 
        initial_state=initial_state, 
        cost_function=cost_function, 
        operations=noon_creation_operations, 
        max_iter=max_iter, 
        initial_guess=initial_guess
    )
    noon_creation_params = results.theta
    noon_creation_params[T4_PARAM_INDEX] = noon_creation_params[T4_PARAM_INDEX] / 4


    # Define new initial state:
    cat_creation_operations = \
        noon_creation_operations + \
        [standard_operations.power_pulse_on_specific_directions(power=1, indices=[0])] + \
        noon_creation_operations

    cat_creation_params = []
    cat_creation_params.extend(noon_creation_params)
    cat_creation_params.append(pi)  # x pi pulse
    cat_creation_params.extend(noon_creation_params)

    # our almost gkp state:
    cat_state = coherent_control.custom_sequence(state=initial_state, theta=cat_creation_params, operations=cat_creation_operations)


    ## Center the cat-state:
    operations = [
        standard_operations.power_pulse_on_specific_directions(power=1, indices=[0,1,2]),
    ]
    def cost_function(final_state:_DensityMatrixType) -> float : 
        observation_mean = np.trace( final_state @ Sp )
        cost = abs(observation_mean)
        return cost
    results = learn_custom_operation(
        num_moments=num_moments, initial_state=cat_state, cost_function=cost_function, operations=operations, max_iter=max_iter, initial_guess=None
    )
    cat_state = results.final_state

    ## Force cat-state to be on the bottom:
    z_projection = np.real(np.trace( cat_state @ Sz ))
    if z_projection>0:
        cat_state = coherent_control.pulse_on_state(cat_state, x=pi)

    ## Aligning with the y axis:
    operations = [
        standard_operations.power_pulse_on_specific_directions(power=1, indices=[2]),
    ]
    def cost_function(final_state:_DensityMatrixType) -> float : 
        observation_mean = np.trace( final_state @ Sx @ Sx )
        cost = observation_mean
        return cost
    results = learn_custom_operation(
        num_moments=num_moments, initial_state=cat_state, cost_function=cost_function, operations=operations, max_iter=max_iter, initial_guess=None
    )
    cat_state = results.final_state


    trial_state = coherent_control.pulse_on_state(cat_state, x=-0.6)
    _wigner(trial_state)

    # visuals.close_all()
    visuals.draw_now()
    for s in np.linspace(0, 0.02, 6):
        gkp = coherent_control.squeezing(trial_state, strength=s, axis=(1,0) )
        _wigner(gkp, title=f"s={s}")


    # visuals.plot_matter_state(cat_state)
    visuals.plot_matter_state(gkp)
    

    
    return results


def main():
    # results = _run_many_guesses()
    results = creating_gkp_algo()


    # Load state:
    state_on_buttom = saveload.load("state_on_buttom")

    num_moments = state_on_buttom.shape[0]-1

    ## Define operations:
    coherent_control = CoherentControl(num_moments=num_moments)
    standard_operations : CoherentControl.StandardOperations = coherent_control.standard_operations(num_intermediate_states=0)

    for s in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]:
        squeezed_state = coherent_control.squeezing(state_on_buttom, strength=s, axis=(0,1))
        qutip.plot_wigner(qutip.Qobj(squeezed_state))
        plt.text(0,0, f"{s}")

    visuals.plot_matter_state(squeezed_state)

    print("End")

if __name__ == "__main__":
    main()
    # _test_learn_pi_pulse(num_moments=4)    
    print("Done.")