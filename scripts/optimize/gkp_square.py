# ==================================================================================== #
# |                                   Imports                                        | #
# ==================================================================================== #
if __name__ == "__main__":
    import pathlib, sys
    sys.path.append(
        str(pathlib.Path(__file__).parent.parent.parent)
    )

# Everyone needs numpy:
import numpy as np
from numpy import pi

# Use our utils:
from utils import strings

# For typing hints:
from typing import (
    Any,
    Tuple,
    List,
    Union,
    Dict,
    Final
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
    learn_custom_operation,
    learn_custom_operation_by_partial_repetitions,
    BaseParamType,
    FreeParam,
    FixedParam,
)

# For physical states:
from physics.gkp import get_gkp_cost_function
from physics.fock import Fock


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
   
def best_sequence_params(
    num_atoms:int, 
    /,*,
    num_intermediate_states:int=0    
)-> Tuple[
    List[BaseParamType],
    List[Operation]
]:
       
    # Define operations:
    coherent_control = CoherentControl(num_atoms=num_atoms)    
    standard_operations : CoherentControl.StandardOperations = coherent_control.standard_operations(num_intermediate_states=num_intermediate_states)
    
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
   
    # theta = [
    #     -0.0014068851683347 , +0.1215726163875156 , -0.3673153331306065 , +0.0249399804630062 , +0.2269478535658498 , 
    #     -0.0190652512267484 , +0.2510723996692263 , -1.0607112513459973 , +0.0024516613121189 , +0.2156517674728328 , 
    #     +0.0319448795412145 , +0.1558487506505765 , -0.0062749313184094 , -0.0039112520369149 , -0.0663501151821770 , 
    #     +2.4499465316584099 , -1.1174755201996209 , +1.4082519226188759 , -0.0483762205467460 , -0.2901126138956479 , 
    #     -2.2320871070585953 , +2.2071586343941316 , -1.5501525031566996 , -1.1049653620985869 , -0.2459256908092352 , 
    #     +0.0586744248216218 , -2.5988976065955574 , -1.0735986268925384 , +0.0287576930899591 , -0.1139451953576366 , 
    #     -0.2675536546291929 , -0.2192988258684723 , +0.0562553082535889 , +0.4029275777194282 , +0.1678248016415093 , 
    #     +1.6503807621359785 , -0.2424417180525995 , -0.4117323968553763 , -0.0077387709781708 , +0.1030606188833836 , 
    #     +0.6584106259075400 , +0.1722046531631806 , +0.3612403708545883 , +0.0450059509382275 , -0.0653746424180170 , 
    #     +0.0560022534150009 , -0.2381812144256178 , +0.0760193203431392 , -0.0819591524260033 , +0.0493671010508360 , 
    #     +0.1628340991897231 , -0.3401516974436361 , -0.3380686055728756 , -0.3092228385753361 , +0.9926172832861413 , 
    #     -0.0109342817022105 , -0.9717446733114196 , +1.0844122307811856         
    # ]  # 98.38 fidelity
    theta = [
        -0.0014078188279650 , +0.1218051907866897 , -0.3771403547985771 , +0.0249356193826340 , +0.2270067306882719 , 
        -0.0193357088385090 , +0.2510599878873806 , -1.0608975361709554 , +0.0024567612752524 , +0.2156600716737191 , 
        +0.0319988915347940 , +0.1560628844755161 , -0.0064247101314180 , -0.0039155900133758 , -0.0663651528832130 , 
        +2.4500398390142815 , -1.1174895095407109 , +1.4081961835095740 , -0.0483585560916238 , -0.2901127398949547 , 
        -2.2320056988442616 , +2.2070671753828455 , -1.5500987219879376 , -1.1049676663529908 , -0.2459302703358617 , 
        +0.0587932662988053 , -2.5988022431760101 , -1.0736276867710544 , +0.0287715726780220 , -0.1139445824659741 , 
        -0.2676618670549614 , -0.2193009148203622 , +0.0562972222465117 , +0.4029097063193843 , +0.1678193666140538 , 
        +1.6503878341850395 , -0.2424798467488802 , -0.4116624135075853 , -0.0077481181318850 , +0.1030546269865189 , 
        +0.6585311507714167 , +0.1722683746503755 , +0.3613346433073044 , +0.0449930927186171 , -0.0653857685574835 , 
        +0.0561138455448691 , -0.2383532065506095 , +0.0760639470826474 , -0.0819657789415995 , +0.0493690068767894 , 
        +0.1628968189739926 , -0.3402311519370093 , -0.3381539367789146 , -0.3092854257161782 , +0.9926028071519823 , 
        -0.0113685049661158 , -0.9717507047984810 , +1.0843898032444632 
    ] # 98.55 fidelity

    operations  = [
        rotation, p2_pulse, rotation, p2_pulse, rotation, p2_pulse, rotation, p2_pulse, rotation, p2_pulse, rotation, p2_pulse, rotation, p2_pulse, rotation, p2_pulse, rotation, p2_pulse, rotation,  p2_pulse, rotation, p2_pulse, rotation
    ]
    
    num_operation_params : int = sum([op.num_params for op in operations])
    assert num_operation_params==len(theta)
    
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
    
    assert len(theta)==len(params_bound)==num_operation_params==len(params_lock)  
    param_config : List[BaseParamType] = []
    for i, (initial_value, bounds, is_locked) in enumerate(zip(theta, params_bound, params_lock)):        
        if is_locked:
            this_config = FixedParam(index=i, value=initial_value)
        else:
            this_config = FreeParam(index=i, initial_guess=initial_value, bounds=bounds, affiliation=None)   # type: ignore       
        param_config.append(this_config)
        
    
    return param_config, operations          
    
    
def main(
    num_total_attempts=2000,
    num_atoms:int=40, 
    max_iter_per_attempt=6*int(1e3),
    max_error_per_attempt=1e-17,
    num_free_params:int|None=35,
    sigma=0.0002
) -> LearnedResults:
        
    # Similar to previous method:
    cost_function = get_gkp_cost_function(num_atoms, form="square")
    initial_state = Fock.ground_state_density_matrix(num_atoms=num_atoms)
    
    # Params and operations:
    param_config, operations = best_sequence_params(num_atoms)

    best_result = learn_custom_operation_by_partial_repetitions(
        # Amount:
        num_attempts=num_total_attempts,
        # Mandatory Inputs:
        initial_state=initial_state,
        cost_function=cost_function,
        operations=operations,
        initial_params=param_config,
        # Huristic Params:
        max_iter_per_attempt=max_iter_per_attempt,
        max_error_per_attempt=max_error_per_attempt,
        num_free_params=num_free_params,
        sigma=sigma,
        log_name="GKP-square "+strings.time_stamp()
    )

    print(best_result)
    return best_result

if __name__ == "__main__":
    results = main()
    print("Done.")