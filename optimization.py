
        
# ==================================================================================== #
# |                                   Imports                                        | #
# ==================================================================================== #

# Everyone needs numpy:
import numpy as np

# import our helper modules
from utils import (
    assertions,
    numpy_tools as np_utils,
    visuals
)
from quantum_states.fock import coherent_state

# For states
from schrodinger_evolution import init_state, Params, CommonStates    

# For plotting results:

# For coherent control
from coherentcontrol import (
    S_mats,
    pulse,
    CoherentControl,
)

# For typing hints:
from typing import (
    Tuple,
    List,
    Union,
)

# for optimization:
from scipy.optimize import minimize, OptimizeResult  # for optimization:   
        
# For measuring time:
import time

# For visualizations:
import matplotlib.pyplot as plt  # for plotting test results:

# For OOP:
from dataclasses import dataclass

# ==================================================================================== #
# |                                  Constants                                       | #
# ==================================================================================== #
OPT_METHOD = 'COBYLA'

# ==================================================================================== #
# |                                    Classes                                       | #
# ==================================================================================== #
@dataclass
class LearnedResults():
    params : np.array
    similarity : float
    state : np.matrix


# ==================================================================================== #
# |                                  Typing hints                                    | #
# ==================================================================================== #
_MatrixType = Union[np.matrix, np.array]

# ==================================================================================== #
# |                               Declared Functions                                 | #
# ==================================================================================== #


def learn_specific_state(initial_state:_MatrixType, target_state:_MatrixType, max_iter:int=4 ) -> LearnedResults:

    # Check inputs:
    for state in [initial_state, target_state]:
        assert len(state.shape)==2
        assert state.shape[0] == state.shape[1]
    assert initial_state.shape[0] == target_state.shape[0]
    
    # Set basic properties:
    matrix_size = state.shape[0]
    max_state_num = matrix_size-1
    coherent_control = CoherentControl(max_state_num)

    # Helper functions:
    def _derive_cost_function(theta:np.array) -> float :  
        final_state = coherent_control.pulse_on_state(initial_state, *theta)
        diff = np.linalg.norm(final_state - target_state)
        cost = diff**2
        return cost

    def _find_optimum()->OptimizeResult:
        initial_point = np.array([0.0]*3)
        options = dict(
            maxiter = max_iter
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

    # Unpack minimization results:    
    theta = opt.x
    assert len(theta)==3
    final_state = coherent_control.pulse_on_state(initial_state, *theta)
    
    # Pack learned-results:
    return LearnedResults(
        params=theta,
        state=final_state,
        similarity=opt.fun
    )
    


# ==================================================================================== #
# |                                Inner Functions                                   | #
# ==================================================================================== #


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
    res = LearnedResults(params=theta, state=rho_final)
    return res


def _learn_pi_pulse_only_x(num_iter:int=4, N:int=2, plot_on:bool=True):
    
    # Define pulse:
    Sx, Sy, Sz = S_mats(N)
    _pulse = lambda x, y, z, c : pulse( x,y,z, Sx,Sy,Sz, c )
    _x_pulse = lambda c : _pulse(1,0,0,c)    

    # init:
    params = Params(N=N)
    rho_initial = init_state(params, CommonStates.Ground)
    rho_target  = init_state(params, CommonStates.FullyExcited)

    # Helper functions:
    def _apply_pulse_on_initial_state(c:float) -> np.matrix: 
        p = _x_pulse(c)
        rho_final = p * rho_initial * p.getH()
        return rho_final

    def _derive_cost_function(c:float) -> float :  
        rho_final = _apply_pulse_on_initial_state(c)
        diff = np.linalg.norm(rho_final-rho_target)
        cost = diff**2
        return cost

    """ 

    cost functions:

    * Even\odd cat states (atomic density matrix)  (poisonic dist. pure state as a |ket><bra| )

    * purity measure:  trace(rho^2)
        1 - if pure
        1/N - maximally not pure 

    * BSV light
    """

    def _find_optimum():
        initial_point = 0.00
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
    c = opt.x
    assert len(c)==1
    assert np.isreal(c)[0]

    rho_final = _apply_pulse_on_initial_state(c)
    np_utils.print_mat(rho_final)

    # visualizing light:
    if plot_on:
        title = f"MAX_ITER={num_iter}"
        plot_city(rho_final, title=title)



# ==================================================================================== #
# |                                  main tests                                      | #
# ==================================================================================== #


def _test_learn_pi_pulse_only_x():
    for num_iter in [1, 2, 5, 10]:
        _learn_pi_pulse_only_x(num_iter=num_iter, plot_on=True)
        visuals.save_figure(file_name=f"learn_pi_pulse_only_x num_iter {num_iter}")

def _test_learn_pi_pulse():
    for num_iter in [1, 2, 5, 10, 20]:
        res = _learn_pi_pulse(num_iter=num_iter, plot_on=True)
        visuals.save_figure(file_name=f"learn_pi_pulse num_iter {num_iter}")

def _test_learn_state(max_fock_num:int=4, plot_on:bool=False):

    assertions.even(max_fock_num)
    
    zero_state = coherent_state(max_num=max_fock_num, alpha=0.00,  type_='normal')
    cat_state  = coherent_state(max_num=max_fock_num, alpha=1.00, type_='normal')
    
    rho_initial = zero_state.to_density_matrix(max_num=max_fock_num)
    rho_target  = cat_state .to_density_matrix(max_num=max_fock_num) 

    if plot_on:
        visuals.plot_city(rho_initial)
        visuals.plot_city(rho_target)

    # np_utils.print_mat(rho_initial)
    # np_utils.print_mat(rho_target)

    results = learn_specific_state(rho_initial, rho_target, max_iter=10000)
    print(results.similarity)

    if plot_on:
        visuals.plot_city(results.state)

    return results


if __name__ == "__main__":
    # _test_learn_pi_pulse_only_x()
    # _test_learn_pi_pulse()
    _test_learn_state()
    print("Done.")