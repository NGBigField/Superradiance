# ==================================================================================== #
# |                                   Imports                                        | #
# ==================================================================================== #
if __name__ == "__main__":
    import pathlib, sys
    sys.path.append(str(pathlib.Path(__file__).parent.parent))

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
        -0.0014080019082643 , +0.1216886741418660 , -0.3767812006535233 , +0.0249391080072267 , +0.2270524199249873 ,
        -0.0190792468053647 , +0.2511478018895302 , -1.0607875909520657 , +0.0024543880414964 , +0.2156543886294543 ,
        +0.0319680296119411 , +0.1559615861132347 , -0.0064186666459340 , -0.0039143610743087 , -0.0663566117116946 ,
        +2.4499997109155585 , -1.1173918864922765 , +1.4082451777926708 , -0.0483591777816203 , -0.2901090011850217 ,
        -2.2321169580697093 , +2.2071198472915405 , -1.5500255283947151 , -1.1049641309627520 , -0.2459233721981643 ,
        +0.0587366195371482 , -2.5988728163492514 , -1.0735059782354361 , +0.0287673619344012 , -0.1139432665624421 ,
        -0.2676532390706430 , -0.2193166936698678 , +0.0563066898566962 , +0.4029194185437229 , +0.1678201947686049 ,
        +1.6503553019042536 , -0.2424906239145306 , -0.4116951547628848 , -0.0077439369607064 , +0.1030566896889450 ,
        +0.6585064675243260 , +0.1722097288111574 , +0.3613044770314864 , +0.0449881143122336 , -0.0653808288900754 ,
        +0.0560601854677509 , -0.2383791133406147 , +0.0760224097139475 , -0.0819641357272693 , +0.0493717925909744 ,
        +0.1628416328258945 , -0.3401656111707854 , -0.3380969024824328 , -0.3092317700816468 , +0.9926142966288561 ,
        -0.0110814466974909 , -0.9717821807715687 , +1.0843858803416779
    ] # 98.54 fidelity

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
    max_iter_per_attempt=30*int(1e3),
    max_error_per_attempt=1e-13,
    num_free_params=None,
    sigma=0.0000002
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