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
    TypeAlias,
    NamedTuple,
)

# import our helper modules
from utils import (
    visuals,
    saveload,
    types,
    decorators,
    strings,
    assertions,
    sounds,
    lists,
    errors,
)

# For coherent control
from coherentcontrol import (
    CoherentControl,
    _DensityMatrixType,
    Operation,
)
        
# For OOP:
from dataclasses import dataclass, field
from enum import Enum, auto

# Import optimization options and code:
from optimization import (
    LearnedResults,
    add_noise_to_params,
    learn_custom_operation,
    ParamLock,
    BaseParamType,
    FreeParam,
    FixedParam,
    CostFunctions,
    _initial_guess,
)
import metrics

# For operations:
import coherentcontrol
from fock import Fock, cat_state

# For managing saved data:
from saved_data_manager import NOON_DATA, exist_saved_noon, get_saved_noon, save_noon

from optimization_and_operations import pair_custom_operations_and_opt_params_to_op_params, free_all_params

# For timing and random seeds:
import time


# ==================================================================================== #
# |                                  Constants                                       | #
# ==================================================================================== #
OPT_METHOD : Final[str] = "Nelder-Mead" #'SLSQP' # 'Nelder-Mead'
NUM_PULSE_PARAMS : Final = 4  

TOLERANCE : Final[float] = 1e-16  # 1e-12
MAX_NUM_ITERATION : Final[int] = int(1e6)  

T4_PARAM_INDEX : Final[int] = 5

# ==================================================================================== #
# |                                Inner Functions                                   | #
# ==================================================================================== #


def _rand(n:int, sigma:float=1)->list:
    return list(np.random.randn(n)*sigma)

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

def _common_4_legged_search_inputs(num_moments:int, num_transition_frames:int=0):
    ## Check inputs:
    assertions.even(num_moments)
    
    ## Define operations:
    initial_state = Fock.excited_state_density_matrix(num_moments)
    coherent_control = CoherentControl(num_moments=num_moments)
    standard_operations : CoherentControl.StandardOperations = coherent_control.standard_operations(num_intermediate_states=num_transition_frames)
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
    rotation_operation = [standard_operations.power_pulse_on_specific_directions(power=1)]


    noon_data = _load_or_find_noon(num_moments)

    # Define target:
    target_4legged_cat_state = cat_state(num_moments=num_moments, alpha=3, num_legs=4).to_density_matrix()
    # visuals.plot_matter_state(target_4legged_cat_state, block_sphere_resolution=200)
    def cost_function(final_state:_DensityMatrixType) -> float : 
        return -1 * metrics.fidelity(final_state, target_4legged_cat_state)   
    
    # Define operations:    
    cat4_creation_operations = \
        noon_creation_operations + \
        rotation_operation + \
        noon_creation_operations + \
        rotation_operation + \
        noon_creation_operations + \
        rotation_operation 
            

    # Initital guess and the fixed params vs free params:
    num_noon_params = 8
    free  = ParamLock.FREE
    fixed = ParamLock.FIXED
    noon_data_params = [val for val in noon_data.params]
    noon_affiliation = list(range(1, num_noon_params+1))
    # noon_affiliation = [None]*num_noon_params
    noon_lockness    = [free]*num_noon_params  # [fixed]*8
    noon_bounds      = [None]*num_noon_params
    rot_bounds       = [(-pi, pi)]*3

    params_value       = noon_data_params + _rand(3)  + noon_data_params + _rand(3)  + noon_data_params + _rand(3)  
    params_affiliation = noon_affiliation + [None]*3  + noon_affiliation + [None]*3  + noon_affiliation + [None]*3  
    params_lockness    = noon_lockness    + [free]*3  + noon_lockness    + [free]*3  + noon_lockness    + [free]*3  
    params_bound       = noon_bounds      + rot_bounds+ noon_bounds      + rot_bounds+ noon_bounds      + rot_bounds
    assert lists.same_length(params_affiliation, params_lockness, params_value, params_bound)

    param_config : List[BaseParamType] = []
    for i, (affiliation, lock_state, initial_value, bounds) in enumerate(zip(params_affiliation, params_lockness, params_value, params_bound)):
        if lock_state == ParamLock.FREE:
            param_config.append(FreeParam(
                index=i, initial_guess=initial_value, affiliation=affiliation, bounds=bounds
            ))
        else:
            param_config.append(FixedParam(
                index=i, value=initial_value
            ))



    return initial_state, cost_function, cat4_creation_operations, param_config

def common_good_starts() -> Generator[list, None, None]:

    for item in [ \
        [ 
            2.78611668e+00,  
            8.78657591e-01, 
            -1.10548169e+01, 
            -9.17436114e-01,
            1.25958016e-01,  
            2.05399498e+00,  
            6.11934061e-02, 
            -1.14385562e+02,
            -7.42116525e-01, 
            2.28624127e+00,  
            1.44418193e-01, 
            -3.10637828e+00,
            2.74037410e+00,  
            3.14159265e+00, 
            -1.80498821e-02, 
            -5.26216960e-01,
            -3.73102342e-01
        ],
        [ 
            1.29715405e+00,  
            9.79621861e-01, 
            -1.18402567e+01,  
            5.85893730e-01,
            4.26152467e-01,  
            1.36222538e+00, 
            -2.23090306e+00,  
            -7.74818090e+01,
            -4.62497765e-02,  
            6.19195011e-03, 
            -2.87869076e-01, 
            -2.07285830e+00,
            3.14159265e+00,  
            2.20534006e+00,  
            6.02986735e-01,  
            9.82102284e-01,
            1.38114814e+00
        ],
        [  
            5.12334483,   
            2.08562615, 
            -14.46979106,  
            -1.75381598,   
            1.3261382,
            1.37128228,  
            -0.62730442,  
            -7.88884155,  
            -0.32050426,   
            3.13317347,
            1.57055123,  
            -1.6264514,    
            1.56699369,   
            3.14159265,  
            -1.56638219,
            -2.81588984,   
            0.82422727
        ],
        [ 
            3.59680480e+00,  
            8.70231489e-01, 
            -1.66371644e+01,  
            1.15964709e+00,
            2.77411784e+00,  
            1.61230528e+00, 
            -2.35460255e+00, 
            -9.05062544e+01,
            7.21027556e-01,  
            1.30767356e-02,  
            1.52088975e+00, 
            -1.75960138e+00,
            1.34089331e+00,  
            1.78832679e+00,  
            9.31994377e-01, 
            -7.45783960e-01,
            -8.12888428e-02
        ],
        [ 
            2.81235479e+00,  
            6.58500630e-01, 
            -1.91032004e+01, 
            -5.02577813e-02,
            7.56818763e-01,  
            2.40146756e+00, 
            -1.70876980e+00, 
            -9.43668349e+01,
            3.14065289e+00,  
            1.35396503e+00, 
            -6.96555278e-01, 
            -1.12360133e+00,
            1.47922973e+00,  
            2.54896639e+00,  
            1.44599870e+00, 
            -3.14159265e+00,
            9.64125752e-01
        ] 
    ]:
        yield item



# ==================================================================================== #
# |                                     main                                         | #
# ==================================================================================== #

class AttemptResult(NamedTuple):
    initial_guess : np.ndarray
    result : LearnedResults
    score : float

def exhaustive_search(
    base_guess          : np.ndarray,
    num_moments         : int = 40,
    num_tries           : int = 100,
    num_iter_per_try    : int = int(5*1e3),
    std                 : float = 0.5,
    plot_on             : bool = True
) -> List[AttemptResult]:

    # base_guess = np.array(
    #         [ 2.68167102e+00,  1.61405534e+00, -1.03042969e+01,  5.98736807e-02,
    #         1.26242432e+00,  1.47234240e+00, -1.71681054e+00, -8.64374806e+01,
    #         4.30847192e-01,  7.88459398e-01, -6.89081116e-02, -2.02854074e+00,
    #         2.23136298e+00,  3.14159265e+00,  3.60804145e-03, -2.18231897e+00,
    #         -5.95372440e-02]
    #    )

    all_attempts : List[AttemptResult] = []
    best_attempt : AttemptResult = AttemptResult(initial_guess=0, result=0, score=10)
    for i in range(num_tries):
        print("searching... "+strings.num_out_of_num(i+1, num_tries))
        try:
            guess = add_noise_to_params(base_guess, std=std)
            res = _exhaustive_try(num_moments=num_moments, initial_guess=guess, num_iter=num_iter_per_try)
        except:
            print(f"Skipping try {i} due to an error.")
            continue
        score = res.score
        crnt_attempt = AttemptResult(initial_guess=guess, result=res, score=res.score)
        all_attempts.append(crnt_attempt)
        if crnt_attempt.score < best_attempt.score:
            best_attempt = crnt_attempt
            _print_progress(crnt_attempt)

    # Save results in a dict format that can be read without fancy containers:
    saved_var = dict(
        best_attempt= types.as_plain_dict(best_attempt),
        all_attempts= [types.as_plain_dict(attempt) for attempt in all_attempts],
        base_guess  = base_guess
    )
    file_name = "exhaustive_search "+strings.time_stamp()
    saveload.save(saved_var, name=file_name)
    print(f"Saving file with name '{file_name}'")
    
    # Plot best result:
    if plot_on:
        visuals.plot_matter_state(best_attempt.result.final_state)

    return all_attempts
    

def _print_progress(crnt_attempt:AttemptResult) -> None:
    print(f"New best result!  Fidelity = {crnt_attempt.score}")
    print(f"Theta = {crnt_attempt.result.theta}")
    print(f"Operation Params = {crnt_attempt.result.operation_params}")
    
    
@decorators.multiple_tries(3)
def _exhaustive_try(num_moments:int, initial_guess:np.ndarray, num_iter:int=MAX_NUM_ITERATION) -> LearnedResults:

    initial_state, cost_function, cat4_creation_operations, param_config = _common_4_legged_search_inputs(num_moments)

    results = learn_custom_operation(
        num_moments=num_moments, 
        initial_state=initial_state, 
        cost_function=cost_function, 
        operations=cat4_creation_operations, 
        max_iter=num_iter, 
        parameters_config=param_config,
        initial_guess=initial_guess
    )
    
    return results


def creating_4_leg_cat_algo(
    num_moments:int=40
) -> LearnedResults:


    initial_guess = add_noise_to_params( 
        np.array(
            [ 2.68167102e+00,  1.61405534e+00, -1.03042969e+01,  5.98736807e-02,
            1.26242432e+00,  1.47234240e+00, -1.71681054e+00, -8.64374806e+01,
            4.30847192e-01,  7.88459398e-01, -6.89081116e-02, -2.02854074e+00,
            2.23136298e+00,  3.14159265e+00,  3.60804145e-03, -2.18231897e+00,
            -5.95372440e-02]
       )
       , std=2.0
    )
        
    initial_state, cost_function, cat4_creation_operations, param_config = _common_4_legged_search_inputs(num_moments)

    results = learn_custom_operation(
        num_moments=num_moments, 
        initial_state=initial_state, 
        cost_function=cost_function, 
        operations=cat4_creation_operations, 
        max_iter=MAX_NUM_ITERATION, 
        parameters_config=param_config,
        initial_guess=initial_guess
    )
    
    
    fidelity = -1 * results.score
    print(f"fidelity is { fidelity }")
    
    operation_params = results.operation_params
    print(f"operation params:")
    print(operation_params)
    
    final_state = results.final_state
    visuals.plot_matter_state(final_state)
    
    
    return results

    
    
def _load_all_search_data() -> Generator[dict, None, None]:
    for name, data in saveload.all_saved_data():
        splitted_name = name.split(" ")
        if splitted_name[0]=="exhaustive_search":
            yield data    
    

def _study():
    # results = exhaustive_search()
    # print("End")

    # search_results = saveload.load("key_saved_data//exhaustive_search 2022.12.31_08.37.25")
    # for attempt in search_results["all_attempts"]:
    #     score = attempt["score"]
    #     theta = attempt["result"]["theta"]
    #     if score < -0.6:
    #         print(theta)

    # all_results = []
    # for initial_guess in common_good_starts():
    #     results = exhaustive_search(
    #         base_guess=np.array(initial_guess),
    #         num_moments=40,
    #         num_tries=5,
    #         num_iter_per_try=int(1e5),
    #         plot_on=False,
    #         std=0.1
    #     )
    #     all_results += results
    # saveload.save(all_results, name="All best results "+strings.time_stamp())

    # print("Done.")
    
    # num_moments = 40
    
    # best_result : dict = {}
    # best_score = -0.7
    # count = 0
    # for data in _load_all_search_data():
    #     for result in data["all_attempts"]:
    #         count += 1
    #         if result["score"] < best_score:
    #             best_result = result
    
    # print(count)
    # print(best_result)
    # initial_guess = best_result["initial_guess"]
    # theta = best_result["result"]["theta"]
    
    
    # array([   3.03467614,    0.93387172,  -10.00699257,   -0.72388404,
    #       0.13744785,    2.11175319,    0.18788428, -118.69022356,
    #      -1.50210956,    2.02098048,   -0.21569011,    3.03467614,
    #       0.93387172,  -10.00699257,   -0.72388404,    0.13744785,
    #       2.11175319,    0.18788428, -118.69022356,   -2.9236711 ,
    #       3.01919738,    3.14159265,    3.03467614,    0.93387172,
    #     -10.00699257,   -0.72388404,    0.13744785,    2.11175319,
    #       0.18788428, -118.69022356,   -0.32642685,   -0.87976521,
    #      -0.83782409])
    
    # initial_state, cost_function, cat4_creation_operations, param_config = _common_4_legged_search_inputs(num_moments)
    
    # for op, params in  pair_custom_operations_and_opt_params_to_op_params(cat4_creation_operations, theta, param_config):
    #     print(op.get_string(params))
    
    
    # results = _exhaustive_try(num_moments=num_moments, initial_guess=theta, num_iter=10)
    # print(results)
    
    
    num_moments = 40

    mode = "record_movie"
    
    # opt_theta = np.array(
    #     [   3.03467614,    0.93387172,  -10.00699257,   -0.72388404,
    #         0.13744785,    2.11175319,    0.18788428, -118.69022356,
    #         -1.50210956,    2.02098048,   -0.21569011,   -2.9236711 ,
    #         3.01919738,    3.14159265,   -0.32642685,   -0.87976521,
    #         -0.83782409])

    opt_theta = np.array(
      [   3.02985656,    0.89461558,  -10.6029319 ,   -0.75177908,
          0.17659927,    2.08111341,    0.30032648, -120.46353087,
         -1.51754475,    1.91694016,   -0.42664783,   -3.13543566,
          2.17021358,    3.14159224,   -0.26865575,   -0.92027109,
         -0.9889859 ])

    if mode=="optimize":

        results : LearnedResults = _exhaustive_try(num_moments=num_moments, initial_guess=opt_theta)        
        visuals.plot_matter_state(results.final_state)
        print(results)
    
    if mode=="record_movie":
        num_transition_frames = 20
        fps = 5


        initial_state, cost_function, cat4_creation_operations, param_config = _common_4_legged_search_inputs(num_moments, num_transition_frames)
        
        operations = []
        theta = []
        for operation, oper_params in  pair_custom_operations_and_opt_params_to_op_params(cat4_creation_operations, opt_theta, param_config):
            print(operation.get_string(oper_params))
            theta.extend(oper_params)
            operations.append(operation)
            
        target_4legged_cat_state = cat_state(num_moments=num_moments, alpha=3, num_legs=4).to_density_matrix()
        def _score_str_func(rho:_DensityMatrixType)->str:
            fidel = metrics.fidelity(rho, target_4legged_cat_state)
            return f"fidelity={fidel}"
        
            
        coherent_control = CoherentControl(num_moments=num_moments)
        movie_config = CoherentControl.MovieConfig(
            active=True,
            show_now=False,
            fps=fps,
            num_transition_frames=num_transition_frames,
            num_freeze_frames=fps,
            bloch_sphere_resolution=200,
            score_str_func=_score_str_func
        )
        final_state = coherent_control.custom_sequence(initial_state, theta=theta, operations=operations, movie_config=movie_config)
        print(_score_str_func(final_state))
    print("Done.")

def disassociate_affiliation()->LearnedResults:
    
    num_moments = 40
    num_transition_frames = 0
    
    # Copy best result with affiliated params:
    initial_state, cost_function, cat4_creation_operations, param_configs = _common_4_legged_search_inputs(num_moments, num_transition_frames)
    best_theta = np.array(
      [   3.02985656,    0.89461558,  -10.6029319 ,   -0.75177908,
          0.17659927,    2.08111341,    0.30032648, -120.46353087,
         -1.51754475,    1.91694016,   -0.42664783,   -3.13543566,
          2.17021358,    3.14159224,   -0.26865575,   -0.92027109,
         -0.9889859 ])
    '''
        ## Get operation params for best results:
        base_params = []
        for _, params in pair_custom_operations_and_opt_params_to_op_params(cat4_creation_operations, best_theta, param_configs):
            for param in params:
                base_params.append(param)        
            
        ## Disassociate affiliated params:
        param_configs : List[ParamConfigBase] 
        for config, value in zip(param_configs, base_params):
            if isinstance(config, FreeParam):
                config.affiliation = None
                config.initial_guess = value
            elif isinstance(config, FixedParam):
                config.value = value 
            print(config)
    '''
    param_configs = free_all_params(cat4_creation_operations, best_theta, param_configs)
     
    # Fix rotation params.
    # for i, param in enumerate( param_configs ): 
    #     print(f"i:{i:2}  {param.bounds}   ")
    #     if param.bounds != (None, None):
    #         param_configs[i] = param.fix()
    
    # Learn:
    results = learn_custom_operation(
        num_moments=num_moments, 
        initial_state=initial_state, 
        cost_function=cost_function, 
        operations=cat4_creation_operations, 
        max_iter=MAX_NUM_ITERATION, 
        parameters_config=param_configs
    )
    print(results)
    return results

    
    
    

def _sx_sequence_params(
    standard_operations:CoherentControl.StandardOperations, sigma:float=0.2
)-> Tuple[
    List[BaseParamType],
    List[Operation]
]:
    
    rotation    = standard_operations.power_pulse_on_specific_directions(power=1, indices=[0, 1, 2])
    p2_pulse    = standard_operations.power_pulse_on_specific_directions(power=2, indices=[0, 1])
    stark_shift = standard_operations.stark_shift_and_rot()
            
        
    _rot_bounds   = lambda n : [(-pi, pi)]*n
    _p2_bounds    = lambda n : [(None, None)]*n
    _stark_bounds = lambda n : [(None, None)]*n
    
    _rot_lock   = lambda n : [False]*n 
    _p2_lock    = lambda n : [False]*n
    _stark_lock = lambda n : [False]*n
    
    # previos_best_values = [
    #     -3.13496905e+00,  6.04779209e-01, -2.97065809e+00,  # rot
    #     -7.21249786e-01,  7.92523986e-02,                   # p2
    #      0.0           ,  0.0           ,  0.0,             # stark-shift
    #     -2.26480360e-01, -3.06453241e+00, -7.77837060e-01,  # rot
    #      1.89698575e-01,  1.44668992e-03,                   # p2
    #      0.0           ,  0.0           ,  0.0,             # stark-shift
    #      1.10893652e+00, -6.32039487e-02,  2.43629268e+00,  # rot
    #      1.39075989e-01, -5.08093640e-03,                   # p2
    #      2.03338557e+00,  3.54986211e-01,  1.23905514e+00   # rot
    # ]
    # previous_best_values = [
    #     -3.14132341e+00,  7.26499599e-01, -2.81640184e+00, 
    #     -7.21249786e-01,  0.0,
    #     -2.24533824e-01, -3.06451820e+00, -8.04970536e-01,
    #      1.89698575e-01,  0.0,
    #     -1.20910238e-03, -1.13211280e-03,  1.22899846e-03,
    #      1.39075989e-01,  0.0,
    #      2.03532868e+00,  3.53830170e-01,  1.23912615e+00
    # ]
    # previous_best_values = [-3.14159265e+00,  7.48816345e-01, -3.14158742e+00, -1.11845072e+00,
    #    -2.54551167e-01, -1.29370443e-04, -3.00227534e+00, -8.38431494e-01,
    #     5.62570928e-02, -1.00562122e-01,  5.38651426e-02,  1.86925119e-02,
    #     1.29525864e-01,  2.92131952e-01,  5.46879499e-02,  2.13122296e+00,
    #     3.05040262e-01,  8.45120583e-01]
    
    # previous_best_values = [
    #     -3.14030032,  1.56964558, -1.23099066, -1.06116912, 
    #     -0.20848513, -0.00737964, -2.79015456, -1.1894669 ,
    #      0.04517585, -0.10835456, -0.03379094, -0.12314313,
    #      0.17918845,  0.31359876,  0.07170169,  2.24711138,  0.36310499,  0.91055266
    # ]
    
    # previous_best_values = [ 
    #     -3.14030032,     1.56964558,     -1.23099066,     -1.06116912,     -0.20848513,
    #     3.61693959e-03, -2.75791882e+00, -1.13001522e+00,  4.37562070e-02,
    #    -1.09556315e-01,  3.00420606e-02, -1.89985052e-01,  1.90603431e-01,
    #     3.08800775e-01,  7.66210890e-02,  2.14303499e+00,  2.61838147e-01,
    #     8.67099173e-01,  7.57309369e-03,  2.43802801e-03, -3.69501207e-03,
    #    -9.91619336e-03,  2.54274091e-02
    # ]
    
    # previous_best_values = [-3.14030032,  1.56964558, -1.23099066, -1.06116912, -0.20848513,
    #     0.00361694, -2.75791882, -1.13001522,  0.04375621, -0.10955632,
    #     0.01501858, -0.18233922,  0.18501545,  0.31482114,  0.07721763,
    #     2.04140464,  0.17904715,  1.29808372,  0.01239275,  0.00664981,
    #    -0.11626285,  0.32799851, -0.14023262]
    
    # previous_best_values = [
    #     -3.11809877e+00,  1.80059370e+00, -1.55462574e+00, -1.08964755e+00,
    #    -2.14198793e-01,  8.51353082e-03, -2.62218220e+00, -1.11777407e+00,
    #     2.17978038e-02, -1.16416204e-01, -1.25477684e-02, -2.92207523e-01,
    #     2.38461118e-01,  3.13539220e-01,  7.49079999e-02,  1.97132229e+00,
    #     4.73772111e-02,  1.23930114e+00,  1.40898647e-02,  7.34119457e-03,
    #    -8.82234915e-02,  3.67593470e-01,  1.61897263e-03,
    #    0.0, 0.0, 0.0, 0.0, 0.0, 
    # ]
    
    # previous_best_values = [
    #     0.0, 0.0, 0.0, 0.0, 0.0, 
    #    -2.73964174e+00,  2.03423352e+00, -1.62928889e+00, -1.08470481e+00,
    #    -2.10877461e-01, -2.80748413e-02, -2.56002461e+00, -1.09108989e+00,
    #     2.29558918e-02, -1.15905236e-01, -7.05754005e-03, -3.56090360e-01,
    #     2.40560895e-01,  3.12555987e-01,  7.45061506e-02,  1.99923603e+00,
    #    -1.49483597e-02,  1.17152967e+00,  1.50856556e-02,  7.67289787e-03,
    #    -1.00619005e-01,  3.27342370e-01,  2.63205029e-02,  7.41929725e-04,
    #     9.55277316e-04,  3.46883173e-03,  4.86919756e-04,  2.12570641e-03
    # ]
    
    # previous_best_values = [ 2.64551298e-02, -1.16495825e-01, -9.04855043e-03, -3.87190166e-02,
    #    -1.77760170e-01, -2.27747555e+00,  2.30208055e+00, -1.65254288e+00,
    #    -1.09125669e+00, -2.13360234e-01, -2.12295400e-02, -2.56579479e+00,
    #    -1.07664641e+00,  2.19823025e-02, -1.16139925e-01, -4.95983723e-03,
    #    -3.62125331e-01,  2.38295456e-01,  3.11773381e-01,  7.49915536e-02,
    #     2.05874179e+00, -6.58375089e-02,  1.25318017e+00,  1.25353393e-02,
    #     3.72244913e-03, -1.94542231e-01,  3.47964229e-01,  6.49519029e-03,
    #     2.06871716e-03,  4.03951412e-03,  2.95672995e-03,  1.47839729e-02,
    #     2.66335759e-02]
    
    # previous_best_values =  [
    #     -6.11337359e-02, -4.48316330e-02, -2.82676017e-02, -3.25561302e-02,
    #    -1.98010475e-01, -2.31842421e+00,  2.36194099e+00, -1.70163429e+00,
    #    -1.08924675e+00, -2.10204448e-01, -2.75792429e-02, -2.55973218e+00,
    #    -1.05334435e+00,  2.12767325e-02, -1.16811652e-01,  4.36041716e-03,
    #    -3.71563640e-01,  2.29741367e-01,  3.13366952e-01,  7.62151375e-02,
    #     1.90917446e+00, -3.62066282e-01,  1.05373414e+00,  1.47731610e-02,
    #     1.99157089e-02, -1.08752683e-01,  2.72017441e-01, -2.72095759e-03,
    #     8.58682076e-04, -1.17743989e-02,  6.01113733e-02,  1.02590587e-01,
    #     3.39742620e-01]
    
    previous_best_values = [
        -7.13588363e-02, -3.80137188e-02, -2.39536167e-02, -3.27949400e-02,
       -1.96901816e-01, -2.31905117e+00,  2.37206637e+00, -1.70061176e+00,
       -1.08833084e+00, -2.10303395e-01, -2.41519298e-02, -2.56865778e+00,
       -1.05087505e+00,  2.14462515e-02, -1.16775193e-01,  4.81643506e-03,
       -3.65193448e-01,  2.27217170e-01,  3.14479557e-01,  7.65798517e-02,
        1.90738399e+00, -4.26370890e-01,  1.05314249e+00,  1.61049232e-02,
        2.31567874e-02, -1.12499156e-01,  2.95158748e-01, -2.82196522e-03,
        5.08932313e-04, -1.38633239e-02,  4.88258818e-02,  9.38795946e-02,
        3.75208008e-01,
        0.0, 0.0, 0.0, 0.0, 0.0
        ]
    
    operations  = [ rotation, p2_pulse, rotation, p2_pulse, rotation, p2_pulse, rotation, p2_pulse, rotation, p2_pulse, rotation, p2_pulse, rotation,  p2_pulse, rotation]
    num_operation_params : int = sum([op.num_params for op in operations])
    assert num_operation_params==len(previous_best_values)
    params_value = lists.add(previous_best_values, _rand(num_operation_params, sigma=sigma))
    
    params_bound = []
    params_lock  = []
    for op in operations:
        n = op.num_params
        if op is rotation:
            params_bound += _rot_bounds(n)
            params_lock  += _rot_lock(n)
        elif op is stark_shift:
            params_bound += _stark_bounds(n)
            params_lock  += _stark_lock(n)
        elif op is p2_pulse:
            params_bound += _p2_bounds(n)
            params_lock  += _p2_lock(n)
        else:
            raise ValueError("Not an option")
    
    assert len(params_value)==len(params_bound)==num_operation_params==len(params_lock)  
    param_config : List[BaseParamType] = []
    for i, (initial_value, bounds, is_locked) in enumerate(zip(params_value, params_bound, params_lock)):        
        if is_locked:
            this_config = FixedParam(index=i, value=initial_value)
        else:
            this_config = FreeParam(index=i, initial_guess=initial_value, bounds=bounds, affiliation=None)   # type: ignore       
        param_config.append(this_config)
        
    # Lock first operations:
    for i in range(14):
        param : FreeParam = param_config[i]
        param_config[i] = param.fix()
    
    return param_config, operations          
    
    
def optimized_Sx2_pulses(num_attempts:int=1, num_runs_per_attempt:int=int(1e5), num_moments:int=40, num_transition_frames:int=0) -> LearnedResults:
    # Similar to previous method:
    _, cost_function, _, _ = _common_4_legged_search_inputs(num_moments, num_transition_frames=0)
    initial_state = Fock.ground_state_density_matrix(num_moments=num_moments)
    
    # Define operations:
    coherent_control = CoherentControl(num_moments=num_moments)    
    standard_operations : CoherentControl.StandardOperations  = coherent_control.standard_operations(num_intermediate_states=num_transition_frames)

    best_result : LearnedResults = None
    best_score = np.inf
    
    for attempt_ind in range(num_attempts):
        print(strings.num_out_of_num(attempt_ind+1, num_attempts))
        
        param_config, operations = _sx_sequence_params(standard_operations, sigma=0.00)
        
        try:            
            results = learn_custom_operation(
                num_moments=num_moments, 
                initial_state=initial_state, 
                cost_function=cost_function, 
                operations=operations, 
                max_iter=num_runs_per_attempt, 
                parameters_config=param_config
            )
        
        except Exception as e:
            errors.print_traceback(e)
        
        else:
            if results.score < best_score:
                print("Score: ",results.score)
                print("Theta: ",results.theta)
                best_result = results
                best_score = results.score
        
    return best_result

if __name__ == "__main__":
    # _study()
    # results = disasseociate_affiliation()
    results = optimized_Sx2_pulses()
    print("Done.")