
        
# ==================================================================================== #
# |                                   Imports                                        | #
# ==================================================================================== #

# Everyone needs numpy and numeric stuff:
import numpy as np
from numpy import pi
from scipy.linalg import expm  # matrix exponential


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
    TypeAlias,
    NamedTuple,
    
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
    lists,
    types,
    logs,
)

# For coherent control
from algo.coherentcontrol import (
    CoherentControl,
    _DensityMatrixType,
    Operation,
)

# for optimization:
from scipy.optimize import minimize, OptimizeResult
        
# For measuring time:
import time

# For OOP:
from dataclasses import dataclass, field
from enum import Enum, auto
from copy import deepcopy



# ==================================================================================== #
# |                                  Constants                                       | #
# ==================================================================================== #
OPT_METHOD : Final[str] = "Nelder-Mead" #'SLSQP' # 'Nelder-Mead'
NUM_PULSE_PARAMS : Final = 4  

DEFAULT_TOLERANCE : Final[float] = 1e-12  
MAX_NUM_ITERATION : Final[int] = int(1e5)  # 1e6 

T4_PARAM_INDEX : Final[int] = 5

# ==================================================================================== #
# |                                    Classes                                       | #
# ==================================================================================== #
@dataclass
class LearnedResults():
    theta               : np.ndarray        = None
    operation_params    : List[float]       = None
    score               : float             = None
    initial_state       : np.matrix         = None
    final_state         : np.matrix         = None
    time                : float             = None
    iterations          : int               = None
    operations          : List[Operation]   = None

    def __repr__(self) -> str:
        np_utils.fix_print_length()
        newline = '\n'
        s = ""
        s += f"score={self.score}"+newline
        s += f"theta={self.theta}"+newline
        s += f"operation_params={self.operation_params}"+newline
        s += f"run-time={self.time}"+newline
        s += np_utils.mat_str_with_leading_text(self.initial_state, text="initial_state: ")+newline       
        s += np_utils.mat_str_with_leading_text(self.final_state  , text="final_state  : ")+newline  
        s += f"num_iterations={self.iterations}"+newline
        s += f"operations: {self.operations}"+newline
        return s
    

class ParamLock(Enum):
    FREE  = auto()
    FIXED = auto()


@dataclass
class BaseParamType():
    index : int
    
    @property
    def lock(self)->ParamLock:
        raise AttributeError("Abstract super class without implementation")

    def get_value(self)->float:
        raise AttributeError("Abstract super class without implementation")

    def set_value(self, value:float)->None:
        raise AttributeError("Abstract super class without implementation")

@dataclass
class FreeParam(BaseParamType): 
    affiliation : int | None
    initial_guess : float | None = None
    bounds : Tuple[float, float] | None = None

    @property
    def lock(self)->ParamLock:
        return ParamLock.FREE
    
    def get_value(self)->float|None:
        return self.initial_guess

    def set_value(self, value:float)->None:
        self.initial_guess = value
    
    def fix(self)->'FixedParam':
        """fix Turn param into a fix param
        """
        fixed_value = self.initial_guess
        if isinstance(fixed_value, float):
            return FixedParam(
                index=self.index,
                value=fixed_value,
            )
        else:
            raise ValueError(f"Must have an initial_guess that can be fixed.")
        

@dataclass
class FixedParam(BaseParamType): 
    value : float

    @property
    def lock(self)->ParamLock:
        return ParamLock.FIXED
    
    def get_value(self)->float:
        return self.value    

    def set_value(self, value:float)->None:
        self.value = value
    
    def free(self)->FreeParam:
        """free Turn param into a free param
        """
        return FreeParam(
            index=self.index,
            affiliation=None,
            initial_guess=self.value
        )

    

class OptimizationParams:

    @dataclass
    class Indices:
        free  : List[int]
        fixed : List[int]

    def _find_index_in_list(i:int, l:List[BaseParamType]) -> BaseParamType:
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

        ordered_list : List[BaseParamType] = []
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
        self.ordered_params : List[BaseParamType] = ordered_list

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

                if param.initial_guess is None:
                    lower_bound = -np.pi if param.bounds[0] is None else param.bounds[0]
                    upper_bound = +np.pi if param.bounds[1] is None else param.bounds[1]
                    val = np.random.uniform(low=lower_bound, high=upper_bound)
                else:
                    val = param.initial_guess
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
        

    def variable_params(self) -> Generator[BaseParamType, None, None]:
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

def _initial_result(initial_state:_DensityMatrixType, initial_theta:List[float], operations:List[Operation], cost_function:Callable[[_DensityMatrixType], float])->LearnedResults:
    num_moments = initial_state.shape[0]-1
    coherent_control = CoherentControl(num_moments)
    initial_best_final_state = coherent_control.custom_sequence(initial_state, theta=initial_theta, operations=operations)
    initial_result = LearnedResults(operation_params=initial_theta, score=cost_function(initial_best_final_state))   
    return initial_result

def _params_str(operation_params:List[float], param_width:int=20) -> str:
    # Constants:
    num_params_per_line : int = 5
    extra_space = 4
    
    # Devide into in iterations::
    params = iter(operation_params)     
    done = False
    param_to_str = lambda x: strings.formatted(x, fill=' ', alignment='<', width=param_width, precision=param_width-extra_space, signed=True)
    first_line = True
    s = ""

    while not done:
        
        for j in range(num_params_per_line):
            try:
                param = next(params)
            except StopIteration:
                done = True
                break
            else:
                s += f"{param_to_str(param)}, "
                    
        if done:
            s = s[:-2]  # remove last ,
        else:
            s += "\n"

        first_line = False
             
    return s
    
        

def add_noise_to_free_params(params:List[BaseParamType], sigma:float)->List[BaseParamType]:
    for i, param in enumerate(params):
        if isinstance(param, FreeParam):
            assert param.initial_guess is not None
            param.initial_guess = param.initial_guess + np.random.normal(1)*sigma
            params[i] = param
    return params

def add_noise_to_vector(x:np.ndarray, std:float=1.0) -> np.ndarray:
    n = np.random.normal(scale=std, size=x.shape)
    y = x + n
    return y
        
        
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
 

def _deal_params_config( 
    num_operation_params:int, 
    positive_indices : np.ndarray,
    parameter_configs: List[Tuple[ParamLock, int|float ]] | List[BaseParamType] | None = None
) -> OptimizationParams :

    def _is_positive(i:int)->bool:
        return i in positive_indices

    def _bounds(i:int) -> Tuple[float, float]:
        if _is_positive(i):
            return (0, None)
        else:
            return (None, None)


    if parameter_configs is None:
        free_params  = [FreeParam( index=i, affiliation=None, bounds=_bounds(i)) for i in range(num_operation_params)  ]
        fixed_params = [ ]
        return OptimizationParams(free_params=free_params, fixed_params=fixed_params, num_operation_params=num_operation_params)
    
    assert isinstance(parameter_configs, list)
    common_type = lists.common_super_class(parameter_configs)

    if issubclass(common_type, BaseParamType):
        def _get_all(lock:ParamLock)->List[BaseParamType]:
            lis = []
            for param in parameter_configs:
                if param.lock is not lock:
                    continue
                if isinstance(param, FreeParam):
                    if param.bounds is None:
                        param.bounds = _bounds(param.index)
                lis.append(param)
            return lis 
        free_params  = _get_all(ParamLock.FREE )
        fixed_params = _get_all(ParamLock.FIXED)

    elif common_type is tuple:
        free_params  = [FreeParam( index=i, affiliation=param[1], bounds=_bounds(i)) for i, param in enumerate(parameter_configs) if param[0]==ParamLock.FREE  ]
        fixed_params = [FixedParam(index=i, value=param[1]) for i, param in enumerate(parameter_configs) if param[0]==ParamLock.FIXED ]
    
    else:
        raise TypeError(f"Expected `parameter_configs` to be a list of types tuples or `ParamConfigBase`. Instead got {common_type} ")

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


# ==================================================================================== #
# |                             Common Cost Functions                                | #
# ==================================================================================== #



# ==================================================================================== #
# |                               Declared Functions                                 | #
# ==================================================================================== #


def learn_single_op(
    initial_state:_DensityMatrixType, op:np.matrix, op_proj_to_minimize:np.matrix
) -> Tuple[
    _DensityMatrixType,
    float
]:

    def pulse_on_state(var:float)->_DensityMatrixType:
        p = np.matrix( expm(1j*op*var) )
        return np.matrix( p @ initial_state @ p.getH() )

    def cost_func(theta:np.ndarray)->float:
        var = theta[0]        
        final_state = pulse_on_state(var)
        x2_proj = np.trace(final_state@op_proj_to_minimize)
        return x2_proj

    result = minimize(cost_func, [0])
    
    var = result.x[0]
    final_state = pulse_on_state(var)
    return final_state, var  # type: ignore

def fix_random_params(params:List[BaseParamType], amount:int)->List[BaseParamType]:
    # copy:
    params = deepcopy(params)
    # Choose in random:
    free_params_indices = [i for i, param in enumerate(params) if isinstance(param, FreeParam)]
    indices_to_fix = np.random.choice(free_params_indices, amount, replace=False) 
    # Apply to copy
    for i, param in enumerate(params):
        if i in indices_to_fix:
            assert isinstance(param, FreeParam)
            params[i] = param.fix()
    return params


def learn_custom_operation(    
    initial_state : _DensityMatrixType,
    operations : List[Operation],
    cost_function : Callable[[_DensityMatrixType], float],
    max_iter : int=100, 
    tolerance : Optional[float] = None,
    initial_guess : Optional[np.ndarray] = None,
    parameters_config : Optional[List[BaseParamType]] = None,
    save_results : bool = True,
    print_interval : int = 20
) -> LearnedResults:

    ## Basic data:
    assert initial_state.shape[0]==initial_state.shape[1]
    num_moments : int = initial_state.shape[0]-1

    num_operation_params = sum([op.num_params for op in operations])
    positive_indices = _positive_indices_from_operations(operations)
    param_config : OptimizationParams = _deal_params_config(num_operation_params, positive_indices, parameters_config)
    if initial_guess is None:
        initial_guess = param_config.initial_guess()
    else:
        assert len(initial_guess)==param_config.num_variables


    # Progress_bar
    prog_bar = strings.ProgressBar(max_iter, "Minimizing: ", print_length=100)    
    @decorators.sparse_execution(skip_num=print_interval, default_results=False)
    def _after_each(xk:np.ndarray) -> bool:
        cost = _total_cost_function(xk)
        operation_params = param_config.optimization_theta_to_operations_params(xk)
        extra_str = f"cost = {cost}"+"\n"+f"{_params_str(operation_params)}"
        prog_bar.next(increment=print_interval, extra_str=extra_str)
        finish : bool = False
        return finish

    ## Optimization Config:
    # Define operations:
    coherent_control = CoherentControl(num_moments)
    def _total_cost_function(theta:np.ndarray) -> float : 
        operation_params = param_config.optimization_theta_to_operations_params(theta)
        final_state = coherent_control.custom_sequence(initial_state, theta=operation_params, operations=operations )
        cost = cost_function(final_state)
        return cost

    options = dict(maxiter = max_iter)   
    bounds = param_config.bounds
    tolerance = args.default_value(tolerance, DEFAULT_TOLERANCE)

    # Run optimization:
    start_time = time.time()
    opt_res : OptimizeResult = minimize(
        _total_cost_function, 
        initial_guess, 
        method=OPT_METHOD, 
        options=options, 
        callback=_after_each, 
        bounds=bounds,
        tol=tolerance    
    )
    finish_time = time.time()
    prog_bar.clear()
    
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



def learn_custom_operation_by_partial_repetitions(
    initial_state:_DensityMatrixType,
    cost_function:Callable[[_DensityMatrixType], float],
    operations:List[Operation], 
    initial_params:List[BaseParamType],
    num_attempts:int=2000, 
    max_iter_per_attempt:int = 10*int(1e3), 
    max_error_per_attempt:Optional[float]=None,
    num_free_params:int|None=20,
    sigma:float = 0.002,
    log_name:str=strings.time_stamp()
)-> LearnedResults:
    
    ## Set logging:
    logger = logs.get_logger(filename=log_name)

    ## Check inputs:
    num_operation_params = sum([op.num_params for op in operations])    
    assert len(initial_params)==num_operation_params

    ## Initital theta
    initial_theta = [param.get_value() for param in initial_params]

    ## Set 0'th iteration:
    best_result = _initial_result(initial_state, initial_theta, operations, cost_function)

    ## Iterate:
    for attempt_ind in range(num_attempts):
        logger.info(f"Iteration: {strings.num_out_of_num(attempt_ind+1, num_attempts)}")
        
        ## Lock random params and add noise to free params::
        num_fix_params = 0 if num_free_params is None else num_operation_params-num_free_params
        params = fix_random_params(initial_params, num_fix_params)

        ## Start with param-values from thr best result and add noise to free params:
        theta = best_result.operation_params
        for i, (param, value) in enumerate( zip(params, theta, strict=True) ):
            param.set_value(value)
            params[i] = param
        params = add_noise_to_free_params(params, sigma)
        
        ## Learn the best theta with these locked params:
        try:            
            results = learn_custom_operation(
                initial_state=initial_state, 
                cost_function=cost_function, 
                operations=operations, 
                max_iter=max_iter_per_attempt, 
                tolerance=max_error_per_attempt,
                parameters_config=params
            )
        except Exception as e:
            s = errors.get_traceback(e)
            logger.warning(s)
            continue

        ## Keep the best result:
        if results.score < best_result.score:
            best_result = deepcopy( results )
            
            logger.info("    *** Best Results: *** ")
            logger.info(f"score: {results.score}")
            logger.info(f"theta: \n{_params_str(results.operation_params)}")
            logger.info("\n")
        

    return best_result

# ==================================================================================== #
# |                                  main tests                                      | #
# ==================================================================================== #


def _test():
    from main_gkp import optimized_Sx2_pulses_by_partial_repetition
    results = optimized_Sx2_pulses_by_partial_repetition()

if __name__ == "__main__":
    _test()
    print("Done.")