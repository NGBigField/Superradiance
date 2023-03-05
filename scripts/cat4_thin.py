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


def _sx_sequence_params(
    num_moments:int
)-> Tuple[
    List[BaseParamType],
    List[Operation]
]:
    
    coherent_control = CoherentControl(num_moments=num_moments)    
    standard_operations : CoherentControl.StandardOperations  = coherent_control.standard_operations(num_intermediate_states=0)
    
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
    # theta = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0218893457569274, 0.7191920866055401, 0.019605744261912264, -0.02502281879878626, 0.14155596176935248, 2.2574095995151096, -1.2234221117580204, 1.4235718671654225, -0.04390285036224292, -0.3038995764286472, -2.0617842805506736, 2.345861866241857, -1.7245106953252414, -1.0947524987426371, -0.23248387901706918, 0.18647738726463614, -2.716468091544212, -1.1825978962680104, 0.04014497727894065, -0.10770179093391521, -0.07347936638442679, -0.17164808412507915, 0.21480542585057338, 0.3913657903936454, 0.1223483093091638, 1.6329201058459644, 0.030497635410195803, -0.27498962101885116, -0.02596820370457454, 0.06360478749470103, 0.2661429997470429, -0.15255776739977395, 0.9595180922240361, 0.024823002842259752, -0.017447338819284106, 0.5066348438594075, -0.044245217700777745, 0.39741466989166474, -0.08627499537501082, 0.010043519067349654, 0.49566358349695794, -0.3491169621902839, -1.3388193210681276, 0.002415068734478643, 0.032566928109088, -0.09585930422102079, 0.30459584778998516, 0.5041789951746705]
    # operations  = [
    #     rotation, p2_pulse, rotation, p2_pulse, rotation, p2_pulse, rotation, p2_pulse, rotation, p2_pulse, rotation, p2_pulse, rotation, p2_pulse, rotation, p2_pulse, rotation,  p2_pulse, rotation, p2_pulse, rotation
    # ]    
    # theta = [
    #     0.0, 0.0, 0.0, 
    #     0.0, 0.0, 
    #     0.0, 0.0, 0.0 
    # ]
    # theta = [
    #     +0.6374879230672365 , +0.0504661313568482 , -0.8879532179138008 , -0.1820134605916127 , +0.1029551807445356 , 
    #     -0.7817389957288317 , -1.7758690191539936 , +1.0289063785742170 
    # ]
    theta = [
        +1.6911753538276657 , +0.4165367678990034 , +1.1596610642766465 , +0.4970010986390708 , +1.1626455201688501 , 
        -0.8365536257598889 , -1.0001921078914235 , +1.3845575396713630 
    ]
    
    operations  = [
        rotation, p2_pulse, rotation
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

    
def optimized_Sx2_pulses_by_partial_repetition(
    num_moments:int=40, 
    num_total_attempts:int=2, 
    num_runs_per_attempt:int=3*int(1e3), 
    max_error_per_attempt:Optional[float]=1e-8,
    num_free_params:int|None=5,
    sigma:float=0.8
) -> LearnedResults:
    
    # Define target:
    target_4legged_cat_state = cat_state(num_moments=num_moments, alpha=3, num_legs=4).to_density_matrix()
    initial_state = Fock.ground_state_density_matrix(num_moments=num_moments)
    def cost_function(final_state:_DensityMatrixType) -> float : 
        return -1 * metrics.fidelity(final_state, target_4legged_cat_state)  
    
    # Define operations:
    param_config, operations = _sx_sequence_params(num_moments)

    best_result = learn_custom_operation_by_partial_repetitions(
        # Mandatory Inputs:
        initial_state=initial_state,
        cost_function=cost_function,
        operations=operations,
        initial_params=param_config,
        # Heuristic Params:
        max_iter_per_attempt=num_runs_per_attempt,
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
    # _study()
    # results = optimized_Sx2_pulses()
    results = optimized_Sx2_pulses_by_partial_repetition()
    print("Done.")