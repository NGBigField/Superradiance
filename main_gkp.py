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
    fix_random_params,
)
import metrics

# For operations:
import coherentcontrol
from fock import Fock, cat_state
from gkp import goal_gkp_state

# For managing saved data:
from saved_data_manager import NOON_DATA, exist_saved_noon, get_saved_noon, save_noon

from optimization_and_operations import pair_custom_operations_and_opt_params_to_op_params, free_all_params




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

# ==================================================================================== #
# |                                     main                                         | #
# ==================================================================================== #
   
def _sx_sequence_params(
    standard_operations:CoherentControl.StandardOperations, 
    sigma:float=0.0, 
    theta:Optional[List[float]]=None,
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
    # theta =  [0.12190991602251308, 0.2060558803562983, 0.00333192026805067, -0.018823030192860937, 0.13613924599412758, 2.20448387686236, -1.2638593058688454, 1.5822947181439018, -0.04289043758340635, -0.30733597273574187, -2.012787117849614, 2.330923681363534, -1.6931582242439163, -1.093306196661781, -0.2342670218254389, 0.19271110554829, -2.704269539818503, -1.1727484129510974, 0.04089162696927812, -0.11783018953211505, -0.04692184935596799, -0.19849927527795627, 0.2215188552036332, 0.37270621811579907, 0.14629986391671346, 1.625498134849523, -0.5221906395490123, 0.11291369948539319, -0.013618615834542741, 0.07443388089616541, 0.33025504244917336, 0.34537006988125374, 0.5326122340565969, 0.010453787010001558, -0.027980438899482547, 0.24852930434056142, -0.05839451837835741, 0.5741412293719343, -0.04810902089655947, 0.009450726522086736, 0.1835552649829172, -0.1813723123585866, -0.3303091631392756]
    # theta = [0.12132860460143442, 0.20627245778218978, 0.00555751712593754, -0.018383132145128097, 0.13575989865142915, 2.204444394201843, -1.2449210343975272, 1.581984572826026, -0.04212221748803353, -0.3111570020245362, -2.012541994334103, 2.3311257140455455, -1.693292229767585, -1.0925019242570382, -0.23410412346166287, 0.19279115814461587, -2.639391656929785, -1.195661817167656, 0.034302513705755655, -0.11728489163384835, -0.047606307034099384, -0.243096340857761, 0.23798385277783346, 0.37327740482882893, 0.1405031460006403, 1.535911583925854, -0.5251501279506179, 0.042903180148174805, -0.005340048503498986, 0.07397597975658522, 0.34938110713962733, 0.3449129313895858, 0.7517300789443699, 0.003466084997210724, -0.028213245105096967, 0.24849043609484178, -0.05786695452160051, 0.3565863092002234, -0.04861769158546667, 0.0164243148341192, 0.24146334775780687, -0.17304489910248183, -0.329783547378488]
    # theta = [-0.00697882253988719, 0.3118947994699536, 0.005515796256980187, -0.018214813819630735, 0.13574286267976485, 2.2608341525860496, -1.2342102724427915, 1.5819349280712118, -0.0422362253600466, -0.31111187350688013, -2.000889178276428, 2.3745212322880933, -1.693261496818282, -1.0925907855474175, -0.23506394799099783, 0.1927715418045734, -2.6770964839346574, -1.1959521817891705, 0.03432441064846286, -0.1173534187889485, -0.1161276456526602, -0.19851382208363078, 0.22402232733127503, 0.3732500755042314, 0.13950746576423428, 1.571423421685492, -0.5116236596206778, 0.042878134032166976, -0.0075643771034948, 0.07374803857201391, 0.34914169727937006, 0.28750761457483454, 0.7348873407118499, 0.003447927553336767, -0.02821551554278348, 0.2483504864233292, -0.014256986022513438, 0.3565586984572986, -0.045533114473701605, 0.01655046767919765, 0.22637377719007123, -0.17305599090105464, -0.3295697807672209]
    theta = [-0.0292277902142156, 0.3119066760642112, 0.008138112824623154, -0.019482443500304765, 0.136388491300244, 2.2503289568191165, -1.23422940886016, 1.581967136866912, -0.042417462749022014, -0.31108880243473336, -2.0009000567179065, 2.378235656269246, -1.6569170585605915, -1.092464610697519, -0.23513068835137318, 0.2038050173269485, -2.6745288619105887, -1.195964402736492, 0.03670684864469924, -0.11737848011211749, -0.11603659287753647, -0.19849551003863697, 0.22405331123871117, 0.37324406373735003, 0.1423788991333206, 1.5714548915966617, -0.5116363692810394, 0.06870999123816453, -0.009192874389011698, 0.07365801054360291, 0.3213210646744191, 0.2874279451585201, 0.7592104567792247, 0.00351222592368487, -0.028162559024349835, 0.299947641242318, 0.007939327025143897, 0.35640708512098385, -0.0455851032540685, 0.01654581615182591, 0.1933615043687551, -0.17302306947382876, -0.35029970324973564]

    operations  = [
        rotation, p2_pulse, rotation, p2_pulse, rotation, p2_pulse, rotation, p2_pulse, rotation, p2_pulse, rotation, p2_pulse, rotation, p2_pulse, rotation,  p2_pulse, rotation
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

def get_gkp_cost_function(num_moments:int)->Callable[[_DensityMatrixType], float]:
    # Define target:
    # target_4legged_cat_state = cat_state(num_moments=num_moments, alpha=3, num_legs=4).to_density_matrix()
    taget_state = goal_gkp_state(num_moments)
    # visuals.plot_matter_state(target_4legged_cat_state, block_sphere_resolution=200)
    def cost_function(final_state:_DensityMatrixType) -> float : 
        return -1 * metrics.fidelity(final_state, taget_state)       

    return cost_function

    
def optimized_Sx2_pulses_by_partial_repetition(
    num_moments:int=40, 
    num_attempts:int=2000, 
    num_runs_per_attempt:int=int(5*1e3), 
    num_free_params:int|None=20,
    sigma:float=0.0002
) -> LearnedResults:
    
    # Constants:
    num_transition_frames:int=0
    
    # Similar to previous method:
    cost_function = get_gkp_cost_function(num_moments)
    initial_state = Fock.ground_state_density_matrix(num_moments=num_moments)
    
    # Define operations:
    coherent_control = CoherentControl(num_moments=num_moments)    
    standard_operations : CoherentControl.StandardOperations  = coherent_control.standard_operations(num_intermediate_states=num_transition_frames)
    
    # Params and operations:
    param_config, operations = _sx_sequence_params(standard_operations)
    base_theta = [param.get_value() for param in param_config]
    results = None
    score:float=np.inf
   
    for attempt_ind in range(num_attempts):
    
        if results is None:
            _final_state = coherent_control.custom_sequence(initial_state, theta=base_theta, operations=operations)
            score = cost_function(_final_state)
            theta = base_theta

        elif results.score < score:
            score = results.score
            theta = results.operation_params
            print(f"score: {results.score}")
            print(f"theta: {list(theta)}")
        
        else: 
            pass
            # score stays the best score
            # theta stays the best theta
            
        param_config, operations = _sx_sequence_params(standard_operations, sigma=sigma, theta=theta, num_free_params=num_free_params)            
        print(strings.num_out_of_num(attempt_ind+1, num_attempts))
        
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
        

    assert results is not None
    return results

if __name__ == "__main__":
    # _study()
    # results = optimized_Sx2_pulses()
    results = optimized_Sx2_pulses_by_partial_repetition()
    print("Done.")