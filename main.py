        
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
    ParamConfigBase,
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


from optimization_and_operations import pair_custom_operations_and_opt_params_to_op_params


# ==================================================================================== #
# |                                  Constants                                       | #
# ==================================================================================== #
OPT_METHOD : Final[str] = "Nelder-Mead" #'SLSQP' # 'Nelder-Mead'
NUM_PULSE_PARAMS : Final = 4  

TOLERANCE : Final[float] = 1e-12  # 1e-12
MAX_NUM_ITERATION : Final[int] = int(5*1e4)  # 1e6 

T4_PARAM_INDEX : Final[int] = 5

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

def _common_4_legged_search_inputs(num_moments:int):
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

    def _rand(n:int)->list:
        return list(np.random.randn(n))
            

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

    param_config : List[ParamConfigBase] = []
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
def _exhaustive_try(num_moments:int, initial_guess:np.ndarray, num_iter:int) -> LearnedResults:

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
    

def main():
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
    
    opt_theta = np.array(
        [   3.03467614,    0.93387172,  -10.00699257,   -0.72388404,
            0.13744785,    2.11175319,    0.18788428, -118.69022356,
            -1.50210956,    2.02098048,   -0.21569011,   -2.9236711 ,
            3.01919738,    3.14159265,   -0.32642685,   -0.87976521,
            -0.83782409])
    
    initial_state, cost_function, cat4_creation_operations, param_config = _common_4_legged_search_inputs(num_moments)
    
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
        num_transition_frames=2,
        num_freeze_frames=2,
        bloch_sphere_resolution=15,
        score_str_func=_score_str_func
    )
    final_state = coherent_control.custom_sequence(initial_state, theta=theta, operations=operations, movie_config=movie_config)
    print(final_state)
    print(_score_str_func(final_state))
    print("Done.")

if __name__ == "__main__":
    main()
    # _test_learn_pi_pulse(num_moments=4)    
    print("Done.")