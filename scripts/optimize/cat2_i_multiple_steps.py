# ==================================================================================== #
# |                                   Imports                                        | #
# ==================================================================================== #

if __name__ == "__main__":
    import pathlib, sys
    sys.path.append(
        pathlib.Path(__file__).parent.parent.parent.__str__()
    )

# Everyone needs numpy:
import numpy as np
from numpy import pi

# For typing hints:
from typing import Optional, Tuple, List

# import our helper modules
from utils import sounds, strings, visuals

# For coherent control
from algo.coherentcontrol import (
    CoherentControl,
    _DensityMatrixType,
)

# Import optimization options and code:
from algo.optimization import (
    LearnedResults,
    learn_custom_operation_by_partial_repetitions,
    FixedParam, 
    FreeParam,
    BaseParamType,
    Operation
)

# Common states and cost functions:
from physics.famous_density_matrices import cat_state, ground_state
from algo.common_cost_functions import fidelity_to_cat

# ==================================================================================== #
# |                                  Constants                                       | #
# ==================================================================================== #

# ==================================================================================== #
# |                                Inner Functions                                   | #
# ==================================================================================== #


def best_sequence_params(
    num_atoms:int,
    /,*,
    num_intermediate_states:int=0    
)-> Tuple[
    List[BaseParamType],
    List[Operation]
]:
    
    coherent_control = CoherentControl(num_atoms=num_atoms)    
    standard_operations : CoherentControl.StandardOperations  = coherent_control.standard_operations(num_intermediate_states=num_intermediate_states)
    
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
    #     +1.6672585088573388 , +0.7966649375214807 , +3.2415926535897932 , +1.5714319595349933 , +1.5701701865717275 , 
    #     -0.0000111217866419 , -0.0000015622659345 , +2.3594798466050158
    # ] # fidelity 0.9918 - 1 step
    # theta = [
    #     +1.6672585088573388 , +0.7966649375214807 , +3.2415926535897932 , +1.5714319595349933 , +1.5701701865717275 ,   #1 
    #     -0.0000111217866419 , -0.0000015622659345 , +2.3594798466050158 , 0.0, 0.0,   #2
    #     +0.0, +0.0, +0.0, 0.0, 0.0, #3
    #     +0.0, +0.0, +0.0, 0.0, 0.0, #4
    #     +0.0, +0.0, +0.0 #4
    # ] # fidelity 0.9918 - 4 steps
    #
    # theta = [
    #     +0.0, +0.0 , +0.0 , +pi/2 , +0 ,   #1 
    #     +0.0, +0.0, +pi/2 , 0.0, 0.0,   #2
    #     +0.0, +pi/2, +0.0, 0.0, 0.0, #3
    #     +0.0, +0.0, +0.0, 0.0, 0.0, #4
    #     +0.0, +0.0, +0.0, 0.0, 0.0, #5
    #     +0.0, +0.0, +0.0, 0.0, 0.0, #6
    #     +0.0, +0.0, +0.0 #  final rotation
    # ] # fidelity 0.9918 - 6 steps
    
    # theta = [
    #     -0.0270826178762894 , +0.0420785292642372 , +1.6552895039760658 , +1.5101402120324945 , -0.0008950464974893 ,
    #     -0.1450707540879570 , +0.2292109730479739 , +1.5551756612995788 , -0.0233353251592546 , +0.0233642260889000 ,
    #     +0.0014409019759466 , +1.1375470015048958 , +0.0108297389140748 , +0.0000000784248237 , +0.0000001242354474 ,
    #     +0.1083037282137895 , +0.1067735081262144 , -0.6376625013026994 , +0.0000002425902640 , +0.0000001562681915 ,
    #     +0.0460796592248603 , +0.0485811088429098 , +0.0169410436835062 , +0.0000005761451667 , +0.0000004278663367 ,
    #     +0.0251369727619576 , +0.1486675404429519 , +0.1559500007126816 , +0.0000001227203463 , +0.0000000310174803 ,
    #     +0.0066005859930512 , +0.0805438764732726 , +0.2188462320291897
    # ] # fidelity 0.84 - 6 steps

    # #  fidelity = 0.976477137132  squeezing = 0.66
    # theta = [
    #     +0.4935717071225814 , +1.7488554910458203 , +2.4081075251579991 , +0.1960856683031602 , +0.0868691825942013 ,
    #     +0.7972821358300305 , +0.1037948476124894 , +2.9860069022070066 , -0.1259184534263386 , +0.0416548691243354 ,
    #     +0.8792154967902378 , +1.0851850619300669 , +0.3040529641594746 , +0.0574856817405069 , +0.0210535691786186 ,
    #     +0.7827130861085472 , -0.3885524067827190 , -0.1380271437296924 , +0.0767853562766987 , -0.1174564141277240 ,
    #     +0.4576062201652379 , +1.2726065155223425 , +1.1641048116223858 , +0.0212355775098413 , +0.0844961998343022 ,
    #     +0.2787371711063600 , +0.9041926789581701 , +1.7690041530910321 , -0.0250755863494547 , +0.0076069776716833 ,
    #     +0.6157074876395792 , +0.6264536407238972 , +0.3242211271688092
    # ]

    # #  Minimizing: [█████████████████████████████████████████████████████.......] 3560/4000 fidelity = 0.979844699795  squeezing = 0.586178854093
    # theta = [
    #     +2.5182877101869803 , +2.2222934605991016 , +2.5048918920244096 , +0.1567194868164671 , +0.0516009035325785 ,
    #     +1.2251850590702782 , +0.2277724596999102 , +2.6991852051569785 , -0.1080071840221321 , +0.0536749046433507 ,
    #     +0.5271848527319873 , +0.8713283856740150 , -0.0978028182167738 , +0.0495426294355147 , +0.0071516285147888 ,
    #     +1.1286692972601555 , -0.4898831361526840 , -0.0692721639279655 , +0.0883261117296493 , -0.1224838259871467 ,
    #     +0.4573175859312585 , +1.3549027475910500 , +1.0539207036362579 , +0.0358147650464850 , +0.0890282679927060 ,
    #     +0.0988791256568105 , +0.7689499034343068 , +1.3700782991575577 , +0.0021097078003447 , +0.0028505830119945 ,
    #     +0.9262283238140345 , +0.8747373815786774 , +0.5635819470809464
    # ]

    # fidelity = 0.992312306468  squeezing = 0.646951683761
    theta = [
        +2.8567573275402358 , +2.3530728677226458 , +2.2439229815723869 , +0.1821548064098444 , +0.0716951667788887 ,
        +1.1530588927182843 , +0.3038442874073793 , +2.4789842872232315 , -0.0710739128659769 , +0.0560833338984745 ,
        +0.6579188063783105 , +1.0725239709669492 , -0.1056646686414356 , +0.1015339259605154 , +0.0261682335557583 ,
        +1.0564150184662844 , -0.5078251013379605 , -0.1469865359536207 , +0.0832787357668058 , -0.1077822220912470 ,
        +0.5860337832017745 , +1.1728652677937685 , +0.9667368248421768 , +0.0396459094821724 , +0.0880697870480867 ,
        -0.0244978653355885 , +0.6448744819710170 , +1.3478740653368053 , +0.0136571295627677 , +0.0185285994375842 ,
        +1.0660199114245268 , +1.0563404413843753 , +0.7897371345515618
    ]
    
    operations  = [
        rotation, p2_pulse
    ] * 6 + [rotation]

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



# ==================================================================================== #
#|                                    Main                                            |#
# ==================================================================================== #

    
def main(
    num_atoms:int=12, 
    num_total_attempts:int=2*int(1e3), 
    max_iter_per_attempt:int=4*int(1e3), 
    max_error_per_attempt:Optional[float]=1e-22,
    num_free_params:int|None=None,
    sigma:float=0.0050,
    initial_sigma:float=0.0100,
    squeezing_cost_factor:float=0.2
) -> LearnedResults:
    
    # Define target:
    initial_state = ground_state(num_atoms=num_atoms)    
    cost_function = fidelity_to_cat(num_atoms=num_atoms, num_legs=2, phase=np.pi/2)
    
    # Define operations:
    param_config, operations = best_sequence_params(num_atoms)

    best_result = learn_custom_operation_by_partial_repetitions(
        # Mandatory Inputs:
        initial_state=initial_state,
        cost_function=cost_function,
        operations=operations,
        initial_params=param_config,
        # Heuristic Params:
        initial_sigma=initial_sigma,
        max_iter_per_attempt=max_iter_per_attempt,
        max_error_per_attempt=max_error_per_attempt,
        num_free_params=num_free_params,
        sigma=sigma,
        num_attempts=num_total_attempts,
        log_name="2-Cat"+strings.time_stamp(),
        squeezing_cost_factor=squeezing_cost_factor
    )

    ## Finish:
    sounds.ascend()
    print(best_result)
    return best_result

if __name__ == "__main__":
    results = main()
    print("Done.")