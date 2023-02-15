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
   
def _sx_sequence_params(
    standard_operations:CoherentControl.StandardOperations, 
    theta:Optional[List[float]]=None
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
   
    # theta = [
    #     -0.0037680229526997 , +0.0000884394271794 , +0.0029514529375288 , +0.0000645917035737 , +0.0080792266104302 ,
    #     -0.1526501435499042 , +0.2247723028640892 , -1.0671008847528898 , -0.0081115648861964 , +0.1760017673065679 ,
    #     +0.0003263508424125 , -0.0002258718745719 , -0.0011485080034991 , -0.0003743388581040 , +0.0018818200909347 ,
    #     +2.3429836657259591 , -1.1300889260889515 , +1.5624183371558296 , -0.0489720317201923 , -0.2920868157006946 ,
    #     -2.2492742876749867 , +2.1933077366199516 , -1.5518493611302087 , -1.1029916310137398 , -0.2439500341665869 ,
    #     +0.0798742350420632 , -2.6193361436376321 , -1.0626458039531070 , +0.0246783835647234 , -0.1153689616076803 ,
    #     -0.2857034895513574 , -0.2062042929409859 , +0.0532085400247058 , +0.4049669365608395 , +0.1696157049560441 ,
    #     +1.7144358519784100 , -0.3793682020535904 , -0.2966345250686039 , -0.0080046415965349 , +0.1030727327062982 ,
    #     +0.5889360604628392 , +0.3787673191711979 , +0.4154382012839655 , +0.0484786865943130 , -0.0620421534101190 ,
    #     +0.0431502523261163 , -0.2217594543256087 , +0.0840110909781451 , -0.0849128000974235 , +0.0477856368279429 ,
    #     +0.1845757936587140 , -0.4076319980583663 , -0.3441313699938813 , -0.3135908022983339 , +0.9919356495110174 ,
    #     +0.0102982992989196 , -0.9539513874542562 , +1.0995187843458072 
    # ]
    # theta = [
    #     -0.0014561804068068 , +0.0029037471408926 , +0.0099969572613470 , +0.0024820692075192 , +0.0208256671651012 ,
    #     -0.1817788873846293 , +0.3121290096547211 , -0.9932150964952097 , -0.0036121804106660 , +0.1702075328508327 ,
    #     +0.0021750136305453 , +0.0034648681541515 , +0.0037110318803730 , -0.0011830973938673 , +0.0029624278803408 ,
    #     +2.4667152793205740 , -1.1275254226619094 , +1.4232052803019424 , -0.0491838716268182 , -0.2900163929805303 ,
    #     -2.2531573649765502 , +2.1837054782252072 , -1.5443652608035079 , -1.1025312509160989 , -0.2440336319211189 ,
    #     +0.0801670675287830 , -2.6050990096914428 , -1.0646672207504506 , +0.0232530747236748 , -0.1158923054051571 ,
    #     -0.2887333798118358 , -0.2209270119933027 , +0.0537484272712125 , +0.4058649326778686 , +0.1706609026467166 ,
    #     +1.7105636785632938 , -0.3646133758651829 , -0.3038378632855127 , -0.0087240683258765 , +0.1018647013636054 ,
    #     +0.5854892921249194 , +0.3802348252026706 , +0.4140779402787197 , +0.0489100827632076 , -0.0639364841321367 ,
    #     +0.0333560485175907 , -0.2157468865729748 , +0.0858100830660118 , -0.0850730702935915 , +0.0497650353406776 ,
    #     +0.1957956032315541 , -0.4182668352754574 , -0.3407675263054232 , -0.3145481189302297 , +0.9920696473937038 ,
    #     +0.0118925726468963 , -0.9588204949173791 , +1.1093731915367313
    # ]
    # theta = [
    #     +0.0036576394466196 , +0.0112940329775400 , +0.0336815248775382 , +0.0102713892853683 , +0.0343535832779569 , 
    #     -0.1536571533754147 , +0.3145457054542681 , -0.8200562596808147 , -0.0026845548872741 , +0.1610891706572649 , 
    #     +0.0100941093900457 , +0.0119878715153187 , +0.0080452815310334 , -0.0015802823263179 , +0.0037216833505059 , 
    #     +2.4730784073832566 , -1.1223457852304137 , +1.4012726752359312 , -0.0492131529457357 , -0.2899622066177626 , 
    #     -2.2521129001369613 , +2.1845211884875910 , -1.5271255164288848 , -1.1024589305339365 , -0.2442913101566750 , 
    #     +0.0791086093025903 , -2.5968865920235329 , -1.0778370782131881 , +0.0230516995923300 , -0.1157445286258147 , 
    #     -0.2882379056045869 , -0.2244744177643750 , +0.0546968226876136 , +0.4063569201362506 , +0.1706913473851285 , 
    #     +1.7074454867226598 , -0.3543049289276272 , -0.3126858210948171 , -0.0092531829002423 , +0.1017860706850642 , 
    #     +0.5833571097270117 , +0.3716133047574598 , +0.4066947606450912 , +0.0492087127259493 , -0.0644825214650559 , 
    #     +0.0298102143009586 , -0.2124892298261119 , +0.0875949908766755 , -0.0851811015740629 , +0.0501883759532305 , 
    #     +0.2002002180301420 , -0.4246003157621967 , -0.3370154644226735 , -0.3146806672729273 , +0.9921730086323099 , 
    #     +0.0105564080945560 , -0.9568101724657248 , +1.1100530617224311
    # ]
    theta = [
        +0.0005427290228356 , +0.1313613774206648 , -0.2666841608529606 , +0.0251527947778494 , +0.2210687331404514 ,
        -0.0785100270022274 , +0.3622881281073956 , -1.0474806660778626 , -0.0004791315689836 , +0.1683428070474974 ,
        +0.0672335562615931 , -0.0306577691002731 , +0.0166649239476977 , -0.0023614632567535 , -0.0198894396779224 ,
        +2.3710557885695418 , -1.1117090205542419 , +1.4205001802178427 , -0.0482336328391670 , -0.2902022629116603 ,
        -2.2470461916441895 , +2.1841575273052705 , -1.5368056008972377 , -1.1040856739635401 , -0.2448247039586955 ,
        +0.0790294505550515 , -2.6121011195262489 , -1.0838622096730433 , +0.0264136192318929 , -0.1147296831458362 ,
        -0.2800910250582198 , -0.2094313043016577 , +0.0619927486546490 , +0.4042047139441712 , +0.1686480386560830 ,
        +1.6758104194200865 , -0.2847484215979247 , -0.3708805188433019 , -0.0087853294250139 , +0.1021007502849815 ,
        +0.6173471787290441 , +0.2766743726764943 , +0.3849782777969265 , +0.0485267896409888 , -0.0644169124589915 ,
        +0.0384489816496858 , -0.2258396095244967 , +0.0866106466136370 , -0.0837678546686292 , +0.0492908742398691 ,
        +0.1904087479194388 , -0.4030098527065602 , -0.3414405870871586 , -0.3123114621546215 , +0.9929915661372795 ,
        -0.0064219911125367 , -0.9607028605421299 , +1.0975622200451001        
    ]

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
    
    


    
def learn_sx2_pulses(
    num_moments:int=40, 
    max_iter_per_attempt=3*int(1e3),
    max_error_per_attempt=1e-9,
    num_free_params=28,
    sigma=0.000002
) -> LearnedResults:
    
    # Constants:
    num_transition_frames:int=0
    
    # Similar to previous method:
    cost_function = get_gkp_cost_function(num_moments, form="square")
    initial_state = Fock.ground_state_density_matrix(num_moments=num_moments)
    
    # Define operations:
    coherent_control = CoherentControl(num_moments=num_moments)    
    standard_operations : CoherentControl.StandardOperations  = coherent_control.standard_operations(num_intermediate_states=num_transition_frames)
    
    # Params and operations:
    param_config, operations = _sx_sequence_params(standard_operations)

    best_result = learn_custom_operation_by_partial_repetitions(
        # Mandatory Inputs:
        initial_state=initial_state,
        cost_function=cost_function,
        operations=operations,
        initial_params=param_config,
        # Huristic Params:
        max_iter_per_attempt=max_iter_per_attempt,
        max_error_per_attempt=max_error_per_attempt,
        num_free_params=num_free_params,
        sigma=sigma
    )

    print(best_result)
    return best_result

if __name__ == "__main__":
    # _study()
    # results = optimized_Sx2_pulses()
    results = learn_sx2_pulses()
    print("Done.")