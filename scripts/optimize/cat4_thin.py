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
from utils import sounds, strings

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

# For operations and cost functions:
from physics.fock import Fock, cat_state
from algo import metrics



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
    #     +1.6911753538276657 , +0.4165367678990034 , +1.1596610642766465 , +0.4970010986390708 , +1.1626455201688501 , 
    #     +0 , +0 , +0 , +0 , +0 , 
    #     -0.8365536257598889 , -1.0001921078914235 , +1.3845575396713630 
    # ]
    # theta = [
    #     +0.4318330313098309 , +1.6226884800739532 , -0.5184544170040373 , +0.4904292546661787 , +1.1579761103653765 , 
    #     +0.2711072924647956 , -1.1455548573063417 , -0.0110646563013583 , +0.0058963857195526 , +0.0093821994475182 , 
    #     -0.9014452124841090 , -1.6577610967809480 , +1.8807033704653549         
    # ]
    # theta = [
    #     +0.3658449972290249 , +0.9257611370387263 , -1.6882308749463719 , +0.4376024809598818 , +1.1897942064568279 , 
    #     +1.1670054840863089 , -1.3422791867903432 , +0.4158322335275012 , +0.0188350187791549 , +0.0213576999852772 , 
    #     -0.0528461850923196 , -1.9429029841060719 , +1.2938917481256631       
    # ]
    # theta = [
    #     +0.3887435406908402 , +0.9104233504317537 , -1.6882308749463719 , +0.4379801564966822 , +1.1907636404558120 ,
    #     0.0 , 0.0 , 0.0 , 0.0 , 0.0 , 
    #     +1.1655035689193887 , -1.3427208539321607 , +0.4163584988465759 , +0.0187488498052232 , +0.0211625884315972 ,
    #     -0.0541798806370485 , -1.9406363397783859 , +1.2944312375423341
    # ]
    # theta = [
    #     0.4707,  0.9475, -2.037, 0.4416,  1.1942,  
    #     0.0367,  0.0239 , 0.05 ,   0.0194 , 0.0042 , 
    #     0.0667 , 0.0,  0.0 , 0.0 , 0.0 ,
    #     1.0667 ,-1.3842,  0.5415 , 0.0365 , 0.0344 ,
    #     -0.0775 ,-1.6467 , 1.3874
    # ] # 4 steps
    # theta = [ 
    #     1.1803e+00,  1.5190e+00, -2.7493e+00,  5.5551e-01,  1.2073e+00,  1.9587e-01,  4.5409e-02,  1.2921e-01,  2.9619e-02,  6.3605e-02,  2.5091e-01,  9.9681e-02,  2.4469e-01,  3.3896e-02, -7.0516e-04,  8.0374e-01, -1.6534e+00,  2.2976e-01,  6.8397e-02,  4.6005e-02,  1.2234e-01, -1.4178e+00,  1.5972e+00
    # ] # 4 steps - fidelity  -0.794306029286899    
    # theta = [
    #     +0.9933824943848713 , +1.7150651757491380 , -2.9026307871326704 , +0.5540113318392044 , +1.2050591782818318 ,
    #     +0.1803706749270632 , +0.0425815980291344 , +0.0812747655435369 , +0.0315183979279744 , +0.0653458700128113 ,
    #     +0.2826136211698640 , +0.0848191040866317 , +0.2924162585962168 , +0.0361543686393795 , -0.0010071035429298 ,
    #     +0.7781859595595566 , -1.6776599354103774 , +0.2664149734193946 , +0.0703463529152947 , +0.0518934647941911 ,
    #     +0.1766526799158567 , -1.3666702808565336 , +1.6194467011948834
    # ] # 4 stps - -0.80000
    # theta = [
    #     +0.9933824943848713 , +1.7150651757491380 , -2.9026307871326704 , +0.5540113318392044 , +1.2050591782818318 ,
    #     +0.1803706749270632 , +0.0425815980291344 , +0.0812747655435369 , +0.0315183979279744 , +0.0653458700128113 ,
    #     +0.00 , +0.00 , +0.00 , +0.00 , -0.00 ,
    #     +0.2826136211698640 , +0.0848191040866317 , +0.2924162585962168 , +0.0361543686393795 , -0.0010071035429298 ,
    #     +0.7781859595595566 , -1.6776599354103774 , +0.2664149734193946 , +0.0703463529152947 , +0.0518934647941911 ,
    #     +0.1766526799158567 , -1.3666702808565336 , +1.6194467011948834
    # ] # 5 stps - -0.80000
    # theta = [
    #     +1.1161201910334415 , +0.9947682822578052 , -2.4621189468445284 , +0.5220929972698001 , +1.1166375104815165 ,
    #     +0.0049684175436791 , +0.0395029514114780 , +0.0478232573404673 , +0.0729715544585917 , +0.1768600852617911 ,
    #     +0.0026095439709786 , -0.0564411418897144 , -0.2491209365954900 , -0.0008482246923307 , -0.0191432843654459 ,
    #     +0.3765808183356960 , -0.1816434628795228 , +0.2796549928652780 , +0.0246277947481982 , -0.0354211517070915 ,
    #     +1.3712010681638649 , -0.9742124709014415 , +1.0564151153577712 , +0.0909193257628908 , +0.0632665075248079 ,
    #     +0.9021957809638379 , -1.0488932699324476 , +1.0268710769364500
    # ]  # 5 steps - 0.84 fidelity
    theta = [
        +0.9057924385704990 , +1.3327476308132198 , -2.8775783920694975 , +0.5239722520689665 , +1.1095262160183874 , 
        +0.0444707592339014 , +0.0092392820126474 , +0.0769813858336804 , +0.0028756661735229 , +0.0085831591466655 , 
        -0.0511208711016559 , +0.0336363160505380 , -0.0473304732760926 , +0.1145325339904932 , +0.2839315040176832 , 
        -0.0828081707374388 , +0.0531960602713813 , +0.0414162774425086 , -0.0434844241904851 , -0.1134015325266683 , 
        +0.0845081940635855 , -1.0536338945944854 , +0.0945531196515775 , +0.0233492997899023 , -0.0420349207162328 , 
        +0.9946354684627158 , -0.7704044115350612 , +1.6535430625167526 , +0.0733761346120948 , +0.0372501900026309 , 
        +0.7566205470305009 , -1.3061130647432289 , +0.7318507824181917
    ]  # 6 steps - 0.9 fidelity


    operations  = [
        rotation, p2_pulse, rotation, p2_pulse, rotation, p2_pulse, rotation, p2_pulse, rotation, p2_pulse, rotation, p2_pulse, rotation
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



# ==================================================================================== #
#|                                    Main                                            |#
# ==================================================================================== #

    
def main(
    num_moments:int=40, 
    num_total_attempts:int=1000, 
    max_iter_per_attempt:int=5*int(1e3), 
    max_error_per_attempt:Optional[float]=1e-13,
    num_free_params:int|None=22,
    sigma:float=0.00005,
    initial_sigma:float=0.00
) -> LearnedResults:
    
    # Define target:
    target_4legged_cat_state = cat_state(num_atoms=num_moments, alpha=3, num_legs=4).to_density_matrix()
    initial_state = Fock.ground_state_density_matrix(num_atoms=num_moments)
    def cost_function(final_state:_DensityMatrixType) -> float : 
        return -1 * metrics.fidelity(final_state, target_4legged_cat_state)  
    
    # Define operations:
    param_config, operations = best_sequence_params(num_moments)

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
        log_name="4-Cat-thin "+strings.time_stamp()
    )

    ## Finish:
    sounds.ascend()
    print(best_result)
    return best_result

if __name__ == "__main__":
    results = main()
    print("Done.")