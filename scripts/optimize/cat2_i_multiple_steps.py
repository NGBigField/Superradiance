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

    #  fidelity = 0.976477137132  squeezing = 2.701398165467
    theta = [
        +0.4935717071225814 , +1.7488554910458203 , +2.4081075251579991 , +0.1960856683031602 , +0.0868691825942013 ,
        +0.7972821358300305 , +0.1037948476124894 , +2.9860069022070066 , -0.1259184534263386 , +0.0416548691243354 ,
        +0.8792154967902378 , +1.0851850619300669 , +0.3040529641594746 , +0.0574856817405069 , +0.0210535691786186 ,
        +0.7827130861085472 , -0.3885524067827190 , -0.1380271437296924 , +0.0767853562766987 , -0.1174564141277240 ,
        +0.4576062201652379 , +1.2726065155223425 , +1.1641048116223858 , +0.0212355775098413 , +0.0844961998343022 ,
        +0.2787371711063600 , +0.9041926789581701 , +1.7690041530910321 , -0.0250755863494547 , +0.0076069776716833 ,
        +0.6157074876395792 , +0.6264536407238972 , +0.3242211271688092
    ]

    #  fidelity = 0.954748675250  squeezing = 0.546365365899
    theta = [
        +0.9833441073531750 , +2.3820237932795543 , +2.4842389413961667 , +0.1479828578050181 , +0.0372685601996679 ,
        +1.5399880877363454 , +0.2590517040067716 , +2.5490523682209147 , -0.0983451391160723 , +0.0378777257707828 ,
        +0.5818835543395644 , +0.7559176873708577 , +0.1144047973011100 , +0.0465548090091873 , +0.0099525808911676 ,
        +1.0398837086139876 , -0.3351826506938800 , -0.0979211033379948 , +0.0821475960511857 , -0.1344878066695223 ,
        +0.4615627004525249 , +1.6210472865775358 , +1.0781930381068592 , +0.0287936956940454 , +0.0759103846247303 ,
        +0.4498774233554131 , +0.8092406996576542 , +1.6721267591090099 , +0.0017208984882339 , -0.0009951817125214 ,
        +0.5797000786718760 , +0.6344154972443625 , +0.2137189280921727
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
    max_error_per_attempt:Optional[float]=1e-20,
    num_free_params:int|None=None,
    sigma:float=0.0050,
    initial_sigma:float=0.0100
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
        log_name="2-Cat"+strings.time_stamp()
    )

    ## Finish:
    sounds.ascend()
    print(best_result)
    return best_result

if __name__ == "__main__":
    results = main()
    print("Done.")