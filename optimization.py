
        
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
    Generator,
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
    indices,
    sounds,
    decorators,
)

# For defining coherent states:
from fock import Fock, cat_state

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
from dataclasses import dataclass, field
from enum import Enum, auto

# for plotting stuff wigner
import qutip
import matplotlib.pyplot as plt

# for accessing old results:
from saved_data_manager import NOON_DATA, exist_saved_noon, get_saved_noon, save_noon

# ==================================================================================== #
# |                                  Constants                                       | #
# ==================================================================================== #
OPT_METHOD : Final = "Nelder-Mead" #'SLSQP' # 'Nelder-Mead'
NUM_PULSE_PARAMS : Final = 4  

TOLERANCE = 1e-10  # 1e-12
MAX_NUM_ITERATION = 1e5  # 1e6 

T4_PARAM_INDEX = 5

# ==================================================================================== #
# |                                    Classes                                       | #
# ==================================================================================== #
@dataclass
class LearnedResults():
    theta : np.array = None
    operation_params : List[float] = None
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
    NEGATIVITY  = auto()
    PURITY      = auto()


class ParamLock(Enum):
    FREE  = auto()
    FIXED = auto()


@dataclass
class ParamConfigBase():
    index : int
    
    @property
    def lock(self)->ParamLock:
        raise AttributeError("Abstract super class without implementation")

@dataclass
class FreeParam(ParamConfigBase): 
    affiliation : int | None
    bounds : Tuple[float, float]

    @property
    def lock(self)->ParamLock:
        return ParamLock.FREE

@dataclass
class FixedParam(ParamConfigBase): 
    value : float

    @property
    def lock(self)->ParamLock:
        return ParamLock.FIXED
    

class OptimizationParams:

    @dataclass
    class Indices:
        free  : List[int]
        fixed : List[int]

    def _find_index_in_list(i:int, l:List[ParamConfigBase]) -> ParamConfigBase:
        for param in l:
            if param.index == i:
                return param
        raise ValueError(f"Index {i} not fount in list")

    def __init__(
        self,
        free_params  : List[FreeParam ],
        fixed_params : List[FixedParam],
        num_operation_params : int
    ) -> None:

        self.indices : OptimizationParams.Indices = OptimizationParams.Indices(
            free  = [param.index for param in free_params ],
            fixed = [param.index for param in fixed_params]
        )

        ordered_list : List[ParamConfigBase] = []
        for i in range(num_operation_params):
            if i in self.indices.free:
                param = OptimizationParams._find_index_in_list(i, free_params)
            elif i in self.indices.fixed:
                param = OptimizationParams._find_index_in_list(i, fixed_params)
            else:
                raise ValueError(f"Couldn't find param with index {i}")
            ordered_list.append(param)
            
        ## Keep data
        self.free_params    : List[FreeParam ]      = free_params
        self.fixed_params   : List[FixedParam]      = fixed_params
        self.ordered_params : List[ParamConfigBase] = ordered_list

    def optimization_theta_to_operations_params(self, theta:np.ndarray) -> List[float]:
        # check inputs:
        assert len(theta)==self.num_variables
        assert isinstance(theta, np.ndarray)

        # Prepare Affiliations of shared values:
        shared_values : Dict[int, float] = {}


        # Construct output:
        values : List[float] = []
        thetas = np.nditer(theta)
        for i, param in enumerate(self.ordered_params):
            if isinstance(param, FixedParam):
                value = param.value

            elif isinstance(param, FreeParam):
                affiliation = param.affiliation

                if affiliation is None:
                    value = next(thetas).item()                

                elif affiliation in shared_values:
                    value = shared_values[affiliation]

                else:
                    value = next(thetas).item()                                
                    shared_values[affiliation] = value

            else:
                raise TypeError(f"Unexpected param type '{type(param)}'")
            
            values.append(value)

    
        # Did we exaust all thetas?
        assertions.depleted_iterator(thetas)
        return np.array(values)

    def initial_guess(self, initial_guess:Optional[np.ndarray]=None) -> np.ndarray :
        # helper nums:
        num_free_params = self.num_variables
        used_affiliations = set()
        
        if initial_guess is not None:  # If guess is given:
            raise NotImplementedError("Not yet")
            assert len(initial_guess) == num_free_params, f"Needed number of parameters for the initial guess is {num_free_params} while {len(initial_guess)} were given"
            if isinstance(initial_guess, list):
                initial_guess = np.array(initial_guess)
            if len(positive_indices)>0:
                assert np.all(initial_guess[positive_indices]>=0), f"All decay-times must be non-negative!"    

        else:  # if we need to create a guess:    
            initial_guess = []
            for param in self.free_params:
                
                # Make sure to use affiliated values only once:
                if param.affiliation is not None:
                    if param.affiliation in used_affiliations:
                        continue
                    else:
                        used_affiliations.add(param.affiliation)
                
                lower_bound = -np.pi/2 if param.bounds[0] is None else param.bounds[0]
                upper_bound = +np.pi/2 if param.bounds[1] is None else param.bounds[1]
                val = np.random.uniform(low=lower_bound, high=upper_bound)
                initial_guess.append(val)   
    
        return np.array(initial_guess)
 
    @property
    def bounds(self) -> List[Tuple[float, float]]:
        return [free_param.bounds for free_param in self.variable_params() ]

    @property
    def affiliations(self) -> set:
        res = set()
        for param in self.free_params:
            if param.affiliation is None:
                continue
            res.add(param.affiliation)
        return res
        

    def variable_params(self) -> Generator[ParamConfigBase, None, None]:
        used_affiliations = set()
        for param in self.free_params:
            if param.affiliation is not None:
                if param.affiliation in used_affiliations:
                    continue
                else:
                    used_affiliations.add(param.affiliation)
            yield param

    @property
    def num_variables(self) -> int:
        num_non_affiliated_params = sum([1 if param.affiliation is None else 0 for param in self.free_params])
        num_unique_affiliations = len(self.affiliations)
        return num_non_affiliated_params + num_unique_affiliations

    @property
    def num_fixed(self) -> int:
        return len(self.fixed_params)




# ==================================================================================== #
# |                                 Helper Types                                     | #
# ==================================================================================== #



# ==================================================================================== #
# |                                Inner Functions                                   | #
# ==================================================================================== #


def _load_or_find_noon(num_moments:int, print_on:bool=True) -> NOON_DATA:
    if exist_saved_noon(num_moments):
        noon_data = get_saved_noon(num_moments)
    else:
        
        ## Define operations:
        coherent_control = CoherentControl(num_moments=num_moments)
        standard_operations : CoherentControl.StandardOperations = coherent_control.standard_operations(num_intermediate_states=0)

        ## Define initial state and guess:
        initial_state = Fock.excited_state_density_matrix(num_moments)
        # Noon Operations:
        noon_creation_operations = [
            standard_operations.power_pulse_on_specific_directions(power=1, indices=[0]),
            standard_operations.stark_shift_and_rot(stark_shift_indices=[1], rotation_indices=[0]),
            standard_operations.stark_shift_and_rot(stark_shift_indices=[] , rotation_indices=[0, 1]),
            standard_operations.stark_shift_and_rot(stark_shift_indices=[1], rotation_indices=[0, 1]),
        ]

        initial_guess = _initial_guess()
        
        ## Learn how to prepare a noon state:
        # Define cost function
        cost_function = CostFunctions.fidelity_to_noon(initial_state)            
        noon_results = learn_custom_operation(
            num_moments=num_moments, 
            initial_state=initial_state, 
            cost_function=cost_function, 
            operations=noon_creation_operations, 
            max_iter=MAX_NUM_ITERATION, 
            initial_guess=initial_guess
        )
        sounds.ascend()
        # visuals.plot_city(noon_results.final_state)
        # visuals.draw_now()
        fidelity =  -1 * noon_results.score
        if print_on:
            print(f"NOON fidelity is { fidelity }")
        
        # Save results:
        noon_data = NOON_DATA(
            num_moments=num_moments,
            state=noon_results.final_state,
            params=noon_results.theta,
            operation=[str(op) for op in noon_creation_operations],
            fidelity=fidelity
        )
        save_noon(noon_data)
        
    return noon_data

def _wigner(state:_DensityMatrixType, title:Optional[str]=None)->None:
    fig, ax = qutip.plot_wigner( qutip.Qobj(state) )
    plt.grid(True)
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

    return [  t_1,     t_2,     delta_1,     t_3,       phi_3,    t_4,        phi_4,    delta_2 ]
    # return [ 2.5066,    0.238 ,  -32.9246,    0.58  ,    0.5366,    2.1576,    0.1602, -107.5689]
    # return [  2.5066,    0.238 ,  -32.9246,    0.58  ,    0.5366,    2.1576/4,    0.1602, -107.5689]



def _coherent_control_from_mat(mat:_DensityMatrixType) -> CoherentControl:
    # Set basic properties:
    matrix_size = mat.shape[0]
    max_state_num = matrix_size-1
    return CoherentControl(max_state_num)

def _deal_params_config( 
    num_operation_params:int, 
    positive_indices : np.ndarray,
    parameter_configs: List[Tuple[ParamLock, int|float ]] = None
) -> OptimizationParams :

    def is_positive(i:int)->bool:
        return i in positive_indices

    def bounds(i:int) -> Tuple[float, float]:
        if is_positive(i):
            return (0, None)
        else:
            return (None, None)

    if parameter_configs is None:
        free_params  = [FreeParam( index=i, affiliation=None, bounds=bounds(i)) for i in range(num_operation_params)  ]
        fixed_params = [ ]

    else:
        free_params  = [FreeParam( index=i, affiliation=param[1], bounds=bounds(i)) for i, param in enumerate(parameter_configs) if param[0]==ParamLock.FREE  ]
        fixed_params = [FixedParam(index=i, value=param[1]) for i, param in enumerate(parameter_configs) if param[0]==ParamLock.FIXED ]

    return OptimizationParams(free_params=free_params, fixed_params=fixed_params, num_operation_params=num_operation_params)
    


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

def _deal_initial_guess_old(num_params:int, initial_guess:Optional[np.array]) -> np.ndarray :
    positive_indices = np.arange(NUM_PULSE_PARAMS-1, num_params, NUM_PULSE_PARAMS)
    return _deal_initial_guess(num_free_params=num_params, initial_guess=initial_guess, positive_indices=positive_indices)
    
def _deal_initial_guess(num_free_params:int, initial_guess:Optional[np.array], positive_indices:np.ndarray) -> np.ndarray:
    if initial_guess is not None:  # If guess is given:
        assert len(initial_guess) == num_free_params, f"Needed number of parameters for the initial guess is {num_free_params} while {len(initial_guess)} were given"
        if isinstance(initial_guess, list):
            initial_guess = np.array(initial_guess)
        if len(positive_indices)>0:
            assert np.all(initial_guess[positive_indices]>=0), f"All decay-times must be non-negative!"    
    else:  # if we need to create a guess:    
        initial_guess = np.random.normal(0, np.pi/2, num_free_params)
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
    initial_guess = _deal_initial_guess_old(num_params, initial_guess)
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
# |                             Common Cost Functions                                | #
# ==================================================================================== #

class CostFunctions():

    @staticmethod
    def fidelity_to_noon(initial_state:_DensityMatrixType) -> Callable[[_DensityMatrixType], float] :
        # Define cost function
        matrix_size = initial_state.shape[0]
        target_state = np.zeros(shape=(matrix_size,matrix_size))
        for i in [0, matrix_size-1]:
            for j in [0, matrix_size-1]:
                target_state[i,j]=0.5

        def cost_function(final_state:_DensityMatrixType) -> float : 
            return (-1) * metrics.fidelity(final_state, target_state)

        return cost_function

    @staticmethod
    def weighted_noon(initial_state:_DensityMatrixType) -> Callable[[_DensityMatrixType], float] :
        # Define cost function
        matrix_size = initial_state.shape[0]

        def cost_function(final_state:_DensityMatrixType) -> float : 
            total_cost = 0.0
            for i, j in indices.all_possible_indices((matrix_size, matrix_size)):
                element = final_state[i,j]
                if (i in [0, matrix_size-1]) and  (j in [0, matrix_size-1]):  # Corners:
                    cost = np.absolute(element-0.5)  # distance from 1/2
                else: 
                    cost = np.absolute(element)  # distance from 0
                total_cost += cost
            return total_cost

        return cost_function

# ==================================================================================== #
# |                               Declared Functions                                 | #
# ==================================================================================== #

def learn_custom_operation(    
    num_moments : int,
    initial_state : _DensityMatrixType,
    operations : List[Operation],
    cost_function : Callable[[_DensityMatrixType], float],
    max_iter : int=100, 
    initial_guess : Optional[np.ndarray] = None,
    parameters_config : List[Tuple[ParamLock, int|float ]] = None,
    save_results : bool=True,
) -> LearnedResults:

    # Progress_bar
    prog_bar = visuals.ProgressBar(max_iter, "Minimizing: ")
    
    skip_num = 10
    @decorators.sparse_execution(skip_num=skip_num, default_results=False)
    def _after_each(xk:np.ndarray) -> bool:
        cost = total_cost_function(xk)
        prog_bar.next(increment=skip_num, extra_str=f"cost = {cost}")
        finish : bool = False
        return finish
    

    num_operation_params = sum([op.num_params for op in operations])
    positive_indices = _positive_indices_from_operations(operations)
    param_config : OptimizationParams = _deal_params_config(num_operation_params, positive_indices, parameters_config)
    initial_guess = param_config.initial_guess(initial_guess=initial_guess)


    ## Optimization Config:
    # Define operations:
    coherent_control = CoherentControl(num_moments)
    def total_cost_function(theta:np.ndarray) -> float : 
        operation_params = param_config.optimization_theta_to_operations_params(theta)
        final_state = coherent_control.custom_sequence(initial_state, theta=operation_params, operations=operations )
        cost = cost_function(final_state)
        return cost

    options = dict(maxiter = max_iter)   
    bounds = param_config.bounds

    # Run optimization:
    start_time = time.time()
    opt_res : OptimizeResult = minimize(
        total_cost_function, 
        initial_guess, 
        method=OPT_METHOD, 
        options=options, 
        callback=_after_each, 
        bounds=bounds,
        tol=TOLERANCE    
    )
    finish_time = time.time()
    prog_bar.close()
    
    # Pack learned-results:
    optimal_theta = opt_res.x
    optimal_operation_params = param_config.optimization_theta_to_operations_params(optimal_theta)
    final_state = coherent_control.custom_sequence(initial_state, theta=optimal_operation_params, operations=operations )
    learned_results = LearnedResults(
        theta = optimal_theta,
        operation_params = optimal_operation_params,
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


def creating_4_leg_cat_algo(
    num_moments:int=40
) -> LearnedResults:

    ## Check inputs:
    assertions.even(num_moments)
    
    ## Define operations:
    initial_state = Fock.excited_state_density_matrix(num_moments)
    coherent_control = CoherentControl(num_moments=num_moments)
    standard_operations : CoherentControl.StandardOperations = coherent_control.standard_operations(num_intermediate_states=0)
    Sp = coherent_control.s_pulses.Sp
    Sx = coherent_control.s_pulses.Sx
    Sy = coherent_control.s_pulses.Sy
    Sz = coherent_control.s_pulses.Sz
    noon_creation_operations : List[Operation] = [
        standard_operations.power_pulse_on_specific_directions(power=1, indices=[0]),
        standard_operations.stark_shift_and_rot(stark_shift_indices=[1], rotation_indices=[0]),
        standard_operations.stark_shift_and_rot(stark_shift_indices=[] , rotation_indices=[0, 1]),
        standard_operations.stark_shift_and_rot(stark_shift_indices=[1], rotation_indices=[0, 1]),
    ]


    noon_data = _load_or_find_noon(num_moments)

    # Define target:
    target_4legged_cat_state = cat_state(num_moments=num_moments, alpha=3, num_legs=4).to_density_matrix()
    # visuals.plot_matter_state(target_4legged_cat_state, block_sphere_resolution=200)
    
    
    # Define operations:    
    cat4_creation_operations = \
        noon_creation_operations + \
        [standard_operations.power_pulse_on_specific_directions(power=1)] + \
        noon_creation_operations + \
        [standard_operations.power_pulse_on_specific_directions(power=1)] + \
        noon_creation_operations + \
        [standard_operations.power_pulse_on_specific_directions(power=1)] + \
        noon_creation_operations + \
        [standard_operations.power_pulse_on_specific_directions(power=1)] 

    def _rand(n:int)->list:
        return list(np.random.randn(n))
            

    # Initital guess and the fixed params vs free params:
    num_noon_params = 8
    free  = ParamLock.FREE
    fixed = ParamLock.FIXED
    noon_data_params = [val for val in noon_data.params]
    noon_affiliation = list(range(1, num_noon_params+1))
    noon_lockness    = [free]*num_noon_params  # [fixed]*8
    noon_value       = list(noon_data.params)
    # noon_lockness[T4_PARAM_INDEX] = free


    param_values       = noon_data_params + [0, 0, 0] + noon_data_params + [0, 0, 0] + noon_data_params + [0, 0, 0] + noon_data_params + [0, 0, 0] 
    params_affiliation = noon_affiliation + [None]*3  + noon_affiliation + [None]*3  + noon_affiliation + [None]*3  + noon_affiliation + [None]*3  
    params_lockness    = noon_lockness    + [free]*3  + noon_lockness    + [free]*3  + noon_lockness    + [free]*3  + noon_lockness    + [free]*3     
    params_value       = noon_value       + _rand(3)  + noon_value       + _rand(3)  + noon_value       + _rand(3)  + noon_value       + _rand(3)  
    assert len(params_affiliation)==len(params_lockness)
    assert len(params_affiliation)==len(param_values)

    param_config = [ \
        (lock_state, affiliation) 
        if lock_state == ParamLock.FREE
        else (lock_state, value) 
        for value, affiliation, lock_state  in zip(param_values, params_affiliation, params_lockness)
    ]
    
    num_variables = 3*4+num_noon_params
        
    ## Learn:
    def cost_function(final_state:_DensityMatrixType) -> float : 
        return -1 * metrics.fidelity(final_state, target_4legged_cat_state)

    results = learn_custom_operation(
        num_moments=num_moments, 
        initial_state=initial_state, 
        cost_function=cost_function, 
        initial_guess=initial_guess,
        operations=cat4_creation_operations, 
        max_iter=MAX_NUM_ITERATION, 
        parameters_config=param_config,
    )
    
    
    fidelity = -1 * results.score
    print(f"fidelity is { fidelity }")
    
    operation_params = results.operation_params
    print(f"operation params:")
    print(operation_params)
    
    final_state = results.final_state
    visuals.plot_matter_state(final_state)
    
    
    

    
    

def main():
    # results = _run_many_guesses()
    results = creating_4_leg_cat_algo()
    print("End")

if __name__ == "__main__":
    main()
    # _test_learn_pi_pulse(num_moments=4)    
    print("Done.")