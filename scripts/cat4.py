# ==================================================================================== #
# |                                   Imports                                        | #
# ==================================================================================== #

if __name__ == "__main__":
    import pathlib, sys
    sys.path.append(str(pathlib.Path(__file__).parent.parent))

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
from algo.coherentcontrol import (
    CoherentControl,
    _DensityMatrixType,
    Operation,
)

# Import optimization options and code:
from algo.optimization import (
    LearnedResults,
    add_noise_to_vector,
    learn_custom_operation,
    learn_custom_operation_by_partial_repetitions,
    ParamLock,
    BaseParamType,
    FreeParam,
    FixedParam,
    CostFunctions,
    _initial_guess,
    fix_random_params,
)

# For operations and cost functions:
from physics.fock import Fock, cat_state
from algo import metrics

# For managing saved data:
from utils.saved_data_manager import NOON_DATA, exist_saved_noon, get_saved_noon, save_noon

from algo.optimization_and_operations import pair_custom_operations_and_opt_params_to_op_params, free_all_params



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
        coherent_control = CoherentControl(num_atoms=num_moments)
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
    coherent_control = CoherentControl(num_atoms=num_moments)
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
            guess = add_noise_to_vector(base_guess, std=std)
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


    initial_guess = add_noise_to_vector( 
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
        
            
        coherent_control = CoherentControl(num_atoms=num_moments)
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
    standard_operations:CoherentControl.StandardOperations, 
    sigma:float=0.0, 
    num_free_params:Optional[int]=None
)-> Tuple[
    List[BaseParamType],
    List[Operation]
]:
    
    rotation    = standard_operations.power_pulse_on_specific_directions(power=1, indices=[0, 1, 2])
    p2_pulse    = standard_operations.power_pulse_on_specific_directions(power=2, indices=[0, 1])
    stark_shift = standard_operations.stark_shift_and_rot()
        
    eps = 0.1    
        
    _rot_bounds   = lambda n : [(-pi-eps, pi+eps)]*n
    _p2_bounds    = lambda n : _rot_bounds(n) # [(None, None)]*n
    _stark_bounds = lambda n : [(None, None)]*n
    
    _rot_lock   = lambda n : [False]*n 
    _p2_lock    = lambda n : [False]*n
    _stark_lock = lambda n : [False]*n
   

    # theta = [0.1230243800839618, 0.6191720299851127, -2.3280344384240303e-07, -0.020563759284078914, 0.1135349628174986, 2.20705196071948, -1.2183340894470418, 1.5799032500057237, -0.0436873903142408, -0.2995503422788831, -2.078190942922463, 2.335714330675413, -1.6935087152480237, -1.094542478123508, -0.22991275654402593, 0.19452686725055338, -2.70221081838102, -1.1752795377491556, 0.03932773593530256, -0.10750609661547705, -0.03991859479109913, -0.20072364056375158, 0.22285496406775507, 0.3743729432388033, 0.11137590080067977, 1.709423376749869, -0.45020803849068647, 0.11283133096297475, -0.013141785459664383, 0.07282695266780875, 0.2946167310023212, 0.3338135564683993, 0.5344263960722166, 0.012467076665257853, -0.03637397049464164, 0.2473014913597948, -0.06283368220768366, 0.5773412763402044, -0.04521543808835432, 0.012247470785197952, 0.18238622202996205, -0.1823704254987203, -0.3945560457085364]
    # theta = [0.12210520552567442, 0.6218420640943056, -2.3629898023750997e-07, -0.020551323193012123, 0.11360037926824519, 2.209084925510073, -1.2174430402921996, 1.5794245379006817, -0.04365400854665902, -0.29950979723088356, -2.0775409218000487, 2.33547789305816, -1.693753541362955, -1.0946549241058354, -0.23005165912185938, 0.19325614593960824, -2.701500514431956, -1.1761831506326956, 0.03945902470506803, -0.10745001650855893, -0.04051542864312936, -0.20025730442576847, 0.22203268190777992, 0.37479703188164293, 0.11171146252932072, 1.7083648790272474, -0.4467844201229475, 0.12181555608546937, -0.013123276943561718, 0.07295184131922022, 0.298444298941394, 0.3365721602761523, 0.5382977058362446, 0.01246926680258384, -0.035789378883378256, 0.25872180133719225, -0.05966203138176516, 0.5946598539430703, -0.044415156579541815, 0.011454470516350446, 0.1760120901005723, -0.18270326738487824, -0.41318383205265685, 0.0007908945930680394, 0.0031963223803733254, -9.038725401938367e-06, 0.006560140760391284, -0.04591704083190651]
    # theta = [0.0218893457569274, 0.7191920866055401, 0.019605744261912264, -0.02502281879878626, 0.14155596176935248, 2.2574095995151096, -1.2234221117580204, 1.4235718671654225, -0.04390285036224292, -0.3038995764286472, -2.0617842805506736, 2.345861866241857, -1.7245106953252414, -1.0947524987426371, -0.23248387901706918, 0.18647738726463614, -2.716468091544212, -1.1825978962680104, 0.04014497727894065, -0.10770179093391521, -0.07347936638442679, -0.17164808412507915, 0.21480542585057338, 0.3913657903936454, 0.1223483093091638, 1.6329201058459644, 0.030497635410195803, -0.27498962101885116, -0.02596820370457454, 0.06360478749470103, 0.2661429997470429, -0.15255776739977395, 0.9595180922240361, 0.024823002842259752, -0.017447338819284106, 0.5066348438594075, -0.044245217700777745, 0.39741466989166474, -0.08627499537501082, 0.010043519067349654, 0.49566358349695794, -0.3491169621902839, -1.3388193210681276, 0.002415068734478643, 0.032566928109088, -0.09585930422102079, 0.30459584778998516, 0.5041789951746705]
    theta = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0218893457569274, 0.7191920866055401, 0.019605744261912264, -0.02502281879878626, 0.14155596176935248, 2.2574095995151096, -1.2234221117580204, 1.4235718671654225, -0.04390285036224292, -0.3038995764286472, -2.0617842805506736, 2.345861866241857, -1.7245106953252414, -1.0947524987426371, -0.23248387901706918, 0.18647738726463614, -2.716468091544212, -1.1825978962680104, 0.04014497727894065, -0.10770179093391521, -0.07347936638442679, -0.17164808412507915, 0.21480542585057338, 0.3913657903936454, 0.1223483093091638, 1.6329201058459644, 0.030497635410195803, -0.27498962101885116, -0.02596820370457454, 0.06360478749470103, 0.2661429997470429, -0.15255776739977395, 0.9595180922240361, 0.024823002842259752, -0.017447338819284106, 0.5066348438594075, -0.044245217700777745, 0.39741466989166474, -0.08627499537501082, 0.010043519067349654, 0.49566358349695794, -0.3491169621902839, -1.3388193210681276, 0.002415068734478643, 0.032566928109088, -0.09585930422102079, 0.30459584778998516, 0.5041789951746705]
    
    operations  = [
        rotation, p2_pulse, rotation, p2_pulse, rotation, p2_pulse, rotation, p2_pulse, rotation, p2_pulse, rotation, p2_pulse, rotation, p2_pulse, rotation, p2_pulse, rotation,  p2_pulse, rotation, p2_pulse, rotation
    ]

    num_operation_params : int = sum([op.num_params for op in operations])
    assert num_operation_params==len(theta)
    params_value = lists.add(theta, _rand(num_operation_params, sigma=sigma))
    
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
        
    #Lock first operations:
    if num_free_params is None:
        num_fixed_params = 0
    else:
        num_fixed_params = num_operation_params-num_free_params
    param_config = fix_random_params(param_config, num_fixed_params)
    assert num_fixed_params == sum([1 if param.lock==ParamLock.FIXED else 0 for param in param_config ])
    
    # Free new params:
    for i in range(5):
        param = param_config[-i-1]
        if isinstance(param, FixedParam):
            param_config[-i-1] = param.free()
    
    return param_config, operations          
    
    
def optimized_Sx2_pulses(num_attempts:int=1, num_runs_per_attempt:int=int(1e5), num_moments:int=40, num_transition_frames:int=0) -> LearnedResults:
    # Similar to previous method:
    _, cost_function, _, _ = _common_4_legged_search_inputs(num_moments, num_transition_frames=0)
    initial_state = Fock.ground_state_density_matrix(num_atoms=num_moments)
    
    # Define operations:
    coherent_control = CoherentControl(num_atoms=num_moments)    
    standard_operations : CoherentControl.StandardOperations  = coherent_control.standard_operations(num_intermediate_states=num_transition_frames)

    best_result : LearnedResults = None
    best_score = np.inf
    
    for attempt_ind in range(num_attempts):
        print(strings.num_out_of_num(attempt_ind+1, num_attempts))
        
        param_config, operations = _sx_sequence_params(standard_operations, sigma=0.000)
        
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
    
    
def optimized_Sx2_pulses_by_partial_repetition(
    num_moments:int=40, 
    num_total_attempts:int=2000, 
    num_runs_per_attempt:int=4*int(1e3), 
    max_error_per_attempt:Optional[float]=1e-8,
    num_free_params:int|None=20,
    sigma:float=0.0002
) -> LearnedResults:
    
    # Constants:
    num_transition_frames:int=0
    
    # Define target:
    target_4legged_cat_state = cat_state(num_moments=num_moments, alpha=3, num_legs=4).to_density_matrix()
    initial_state = Fock.ground_state_density_matrix(num_atoms=num_moments)
    def cost_function(final_state:_DensityMatrixType) -> float : 
        return -1 * metrics.fidelity(final_state, target_4legged_cat_state)  
    
    # Define operations:
    coherent_control = CoherentControl(num_atoms=num_moments)    
    standard_operations : CoherentControl.StandardOperations  = coherent_control.standard_operations(num_intermediate_states=num_transition_frames)
    param_config, operations = _sx_sequence_params(standard_operations)

    best_result = learn_custom_operation_by_partial_repetitions(
        # Mandatory Inputs:
        initial_state=initial_state,
        cost_function=cost_function,
        operations=operations,
        initial_params=param_config,
        # Huristic Params:
        max_iter_per_attempt=num_runs_per_attempt,
        max_error_per_attempt=max_error_per_attempt,
        num_free_params=num_free_params,
        sigma=sigma,
        num_attempts=num_total_attempts
    )

    ## Finish:
    sounds.ascend()
    print(best_result)
    return best_result

if __name__ == "__main__":
    # _study()
    # results = optimized_Sx2_pulses()
    results = optimized_Sx2_pulses_by_partial_repetition()
    print("Done.")