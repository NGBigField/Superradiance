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
import metrics

# For operations:
import coherentcontrol
from fock import Fock, cat_state
from gkp import get_gkp_cost_function

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
    # theta = [-0.0292277902142156, 0.3119066760642112, 0.008138112824623154, -0.019482443500304765, 0.136388491300244, 2.2503289568191165, -1.23422940886016, 1.581967136866912, -0.042417462749022014, -0.31108880243473336, -2.0009000567179065, 2.378235656269246, -1.6569170585605915, -1.092464610697519, -0.23513068835137318, 0.2038050173269485, -2.6745288619105887, -1.195964402736492, 0.03670684864469924, -0.11737848011211749, -0.11603659287753647, -0.19849551003863697, 0.22405331123871117, 0.37324406373735003, 0.1423788991333206, 1.5714548915966617, -0.5116363692810394, 0.06870999123816453, -0.009192874389011698, 0.07365801054360291, 0.3213210646744191, 0.2874279451585201, 0.7592104567792247, 0.00351222592368487, -0.028162559024349835, 0.299947641242318, 0.007939327025143897, 0.35640708512098385, -0.0455851032540685, 0.01654581615182591, 0.1933615043687551, -0.17302306947382876, -0.35029970324973564]
    # theta = [-0.015885789633692634, 0.311716733926922, 0.012514583517170823, -0.016228209889783643, 0.1450279532817031, 2.2600129799304502, -1.183044562485391, 1.581871129408326, -0.04175142715498745, -0.31063043862594936, -2.016636676789754, 2.3668009844481688, -1.6539286010448018, -1.0932887547523849, -0.23507008616576852, 0.20394419767436683, -2.674420628945691, -1.196005921262974, 0.03662947630373037, -0.11773728420057664, -0.11567244134936834, -0.1984022034388194, 0.2237499429262888, 0.37279324333170394, 0.14271660261604563, 1.5586075704670468, -0.4431497053410871, 0.00820112142984012, -0.009283388409999538, 0.07386412324958369, 0.3481835058154933, 0.19268320983358617, 0.7593882816083115, 0.003478427723617808, -0.027997276150179065, 0.2998298773769055, 0.008063055059840861, 0.3559681675809572, -0.045404941309051663, 0.016531837598581078, 0.19306895218625963, -0.1525903086592657, -0.3503171248762936]
    # theta = [-0.03419971486778499, 0.36322327109175856, -0.002227562631920986, -0.015732456497068247, 0.15224444655892738, 2.278899076406936, -1.158973166798957, 1.5605336615985985, -0.04091017920834179, -0.31177833182659265, -2.017088214441709, 2.3722305792377956, -1.6598611663308582, -1.093263375724594, -0.23453859680509803, 0.2100544962548183, -2.6809717343993986, -1.1915993902447273, 0.03803938534013472, -0.11650908446322117, -0.12677027055965812, -0.1887973164614425, 0.22703403000707606, 0.3711777253338373, 0.13922688627434066, 1.5854831719808016, -0.43376853555834416, 0.001120667065951536, -0.009195616760030632, 0.07414929650590096, 0.3209550589698724, 0.20051844767955712, 0.7566844373553123, 0.003488889120683073, -0.026985676966745163, 0.33043838059048514, 0.008785710526327206, 0.3572835614670199, -0.04688215368700052, 0.012815291354073728, 0.1806930926337323, -0.15253648492263916, -0.37079662553003223]
    # theta = [-0.06789025150917438, 0.661511185448504, -0.009456326901819533, -0.017002777450465075, 0.16837197323063846, 2.54740725943941, -1.2239955806428486, 1.1559085108720626, -0.041472451791509934, -0.3066390239979153, -2.019990695768935, 2.353739274597255, -1.7333840216557785, -1.0939058036485936, -0.23464130423840432, 0.24948974059986245, -2.7776535804618314, -1.1141855197628685, 0.04061556455713138, -0.11379720380761205, -0.14946288085934137, -0.11571142400933657, 0.20973290342440698, 0.3734783697479107, 0.13278624684949816, 1.6559539852103615, -0.46521205727232906, 0.006531064975896008, -0.014429850850973425, 0.08282528467826628, 0.4445979311364126, 0.3105680370689784, 0.7216489059966745, 0.005324159547369081, -0.026573741278271303, 0.4406587177060125, -0.1620972583141594, 0.2998106456772889, -0.056937689960495266, 0.008489374470194246, 0.07976235961437769, -0.1577468627024814, -0.37777356843152166]
    # theta = [-0.05193908918185616, 0.650607284069566, -0.010792944378829269, -0.007377987003602421, 0.19469349666523084, 2.458111978799339, -1.0389082168033155, 1.4061732703330456, -0.03988684128858472, -0.30865937207539806, -2.0238670102419123, 2.373709851860763, -1.7242887695681781, -1.0959271393235466, -0.23575319501647385, 0.20940368703426504, -2.74525403903281, -1.1525939461210934, 0.04110025441962069, -0.11355831907056213, -0.16343063789824347, -0.1337024443576918, 0.20843006321620222, 0.37537089852661065, 0.13385088167853848, 1.6727022347704206, -0.4913450416682362, 0.0052014235766723455, -0.016140575663271137, 0.08927990422044055, 0.39774830713182574, 0.35304387109413093, 0.7292930724767375, 0.011623297044635234, -0.025334172955155956, 0.42124027605648295, -0.1745058390423399, 0.33226164542042913, -0.0645135868064255, 0.005824116745772148, 0.1368245366928008, -0.133301730519441, -0.3994371875867735]
    # theta = [-0.04124349919137237, 0.6298047321586997, 0.016380500188992834, -0.007682679533853217, 0.19560348427446247, 2.4315959558060305, -1.035692853984667, 1.4410857343984387, -0.03924099801162268, -0.30891314201632025, -2.0235348817480627, 2.3732075756810955, -1.7299753546122094, -1.0961067816967178, -0.23563961690342916, 0.1936048729320392, -2.7280651018592965, -1.1585276613698121, 0.04185404837450546, -0.1129992929113608, -0.16002519543043744, -0.1487805761392881, 0.20420830300938234, 0.37392330411435004, 0.13310439129274915, 1.6789923244654532, -0.5156388624487825, 0.02697922168232084, -0.015004472608208114, 0.0873389853946119, 0.37254031755784045, 0.4090932808985732, 0.7412203468772487, 0.011393811811123145, -0.02509910730905724, 0.4004603234016437, -0.1645116779489389, 0.3068393365344253, -0.06475420581256953, 0.004815251040718315, 0.1691395739652454, -0.17037100182132214, -0.4053560893836458]
    # theta = [-0.03927454103138546, 0.6064543213613001, 0.01750449251865587, -0.00731454743687405, 0.19561834900942382, 2.4102628399798895, -1.0268982280713752, 1.4704611420586908, -0.03922246003115119, -0.3089576808279685, -2.023941005021703, 2.3729652096554856, -1.7283298753873626, -1.0962346506184746, -0.23568374617016002, 0.19411545885214748, -2.728333843808196, -1.160307854038832, 0.04173405690650858, -0.11313141096465498, -0.16078007599591132, -0.1476782325900009, 0.20421777084939863, 0.3741350557077143, 0.13336950466841918, 1.6789410126126811, -0.5139345814726461, 0.027251603993176776, -0.015235690959653277, 0.08735196206303009, 0.3766215894236694, 0.4074454634860104, 0.7423444074156211, 0.011761843275294677, -0.025059000309191296, 0.38976311936350727, -0.16592488367125785, 0.30716971710713337, -0.06476211880768792, 0.005230271085755295, 0.1755221781964157, -0.17178977869154488, -0.4060559065809677]
    
    # Changed the goal function:
    # theta = [-0.03665510659548553, 0.3516383531212155, 0.006243579101451657, -0.0029059258546636326, 0.2614313271223422, 2.1221088051830916, -0.9282417747369314, 1.6673967544482946, -0.044800151541580596, -0.30899855697148015, -2.071293840612629, 2.3716483823700205, -1.8938164053059459, -1.1118572231327968, -0.23702975068661503, 0.14994413040476395, -2.742258671479444, -0.9513759820080065, 0.03311461573596047, -0.11114779023327881, -0.2789387228378284, -0.15901535361101243, 0.18898041978517255, 0.3888728296066507, 0.14384375584107623, 1.775406470148884, -0.3524471316806824, 0.035346498357694725, -0.017626262739562998, 0.07375577281457787, 0.4910065622080446, 0.378887836831868, 0.6607879422769665, 0.009887149925677397, -0.03057738538937138, 0.32283950981817944, -0.19804288290397298, 0.25152012968221715, -0.052298663255873665, 0.006261924733811503, 0.2381437613265205, -0.2335144806373668, -0.4548244233255616]
    # theta = [-0.04453430745221538, 0.5091229504611134, 0.025932841564801915, 0.002433974129222957, 0.27453392088152073, 2.276022429028855, -0.8993235182730293, 1.561925403716406, -0.04280555832129976, -0.30120392925866335, -2.1222247037125594, 2.328089025243405, -1.8895200846417883, -1.1191105450650096, -0.23796399358860937, 0.12533855002362126, -2.6985295173543085, -0.9470008114302151, 0.03478652749203956, -0.11150719973428339, -0.2704371301807325, -0.19986396775305304, 0.21856385211593077, 0.3924414934361413, 0.14944399273944453, 1.6709092689272476, -0.2874497211298391, -0.3036191058993629, -0.012864046365815254, 0.09555596794667348, 0.4873110805060569, 0.357956258742577, 0.46618750658969416, 0.03573623439948531, -0.0381994314480942, 0.07841190333178505, -0.24394373551729992, 0.19229518349901778, -0.07233061570868529, 0.01841307553634603, 0.4032799128109158, -0.40561939801348046, -0.4117617140103246]
    # theta = [-0.04448235472666291, 0.5150292280558915, 0.026079452668245337, 0.002371681201324749, 0.27442952837929124, 2.274968244292623, -0.9000861528847197, 1.5619211041721393, -0.04285463088221836, -0.3013383095716631, -2.1225187907744116, 2.335085498404837, -1.8803457547063451, -1.1190676191552624, -0.23814165750432864, 0.1252796955749016, -2.696955174906911, -0.9501412636480471, 0.03533950802530224, -0.11140626321477753, -0.2701537872109435, -0.20132014998783948, 0.21858527638355862, 0.39228174609004396, 0.14930551038149276, 1.6705573150046038, -0.28631572958074836, -0.3049404419940538, -0.01284023828770909, 0.0955561733935146, 0.4897806476937807, 0.36125951102502973, 0.46644583428113906, 0.03588531624107602, -0.03812781134260948, 0.07784397830592218, -0.2463298200308882, 0.19078174887620902, -0.07210075984394132, 0.018505329229782173, 0.40166146756661025, -0.4096215451952868, -0.41744894151213696]
    # theta =  [0.0709996333941006, 0.4630369534708614, 0.06443662171346384, 0.0034785411225331927, 0.251970189881266, 2.314966271353146, -0.8996826413576425, 1.578815578635789, -0.04099550168015468, -0.29696332620172805, -2.149996808310577, 2.3226303934644825, -1.8390695726294854, -1.121144610134576, -0.24020892044455228, 0.1294682955582539, -2.6949980314489457, -0.9631016612063302, 0.03652176977658232, -0.1094499521511643, -0.2838177634542084, -0.16599041837391404, 0.17724910159823754, 0.3893977577390746, 0.14637034525610446, 1.6766320444932266, -0.35023004148432335, -0.2504490593677237, -0.009759387732651937, 0.09819047935119724, 0.47456950288816213, 0.5086409285693683, 0.4302361893628138, 0.04299174529524075, -0.05063299780703933, 0.06320932952610013, -0.33699273212852154, 0.1881071788863869, -0.07540626312537838, 0.030965393268969764, 0.34434256538389224, -0.43148327372042294, -0.3258300516441791]
    # theta = [0.059986205461964884, 0.44397591665117336, -0.00551241403296313, 0.004341027110217361, 0.2513758895044367, 2.31110084582786, -0.8981754411612041, 1.5947200826181964, -0.04051483982915391, -0.29533103947151906, -2.1488166603205485, 2.3218785863695466, -1.8357331084656168, -1.122321910968175, -0.24053866053597334, 0.12524957171832313, -2.6895130032945476, -0.9752220850501103, 0.03618546951182439, -0.10915640888600979, -0.2932386572870075, -0.16998917797165336, 0.18357917605780505, 0.389613362159819, 0.1475835782569646, 1.6639149162105165, -0.28358544205068004, -0.33258502061129713, -0.007279700422581416, 0.10080595954778555, 0.48230937944541075, 0.4851071904193308, 0.37538921101283473, 0.04232862836557824, -0.05036939628873942, 0.07389295388659387, -0.3737947060172051, 0.19316665984085346, -0.07229822032187608, 0.03264683721820771, 0.34409306639829595, -0.4534686882672394, -0.32102887943456404]
    # theta = [0.06587039788987023, 0.46696763150274845, -0.0022851456692448225, 0.0041273261576339135, 0.2498095716925342, 2.3349335262055027, -0.9049168112099828, 1.5727384672047307, -0.04065222091405338, -0.29516809759800233, -2.151620481629741, 2.3186207076495657, -1.832238253282056, -1.1226160084155854, -0.24105013276148668, 0.12679781586607564, -2.689241038134708, -0.9810122425666121, 0.03609588582477037, -0.10911840233061904, -0.2953802463668197, -0.16733211781272267, 0.18374605156789015, 0.3908515689164125, 0.14868022840340295, 1.6633453795695043, -0.25982719374215224, -0.3487884261357942, -0.008033511746934261, 0.10008872804276256, 0.5117647416892432, 0.4422750184416062, 0.37746876609704594, 0.04271929730776744, -0.05202357217559858, 0.08131022851419054, -0.3643400646074111, 0.17008291533553904, -0.07271049387040773, 0.034557145495543784, 0.3048686175042411, -0.44153657435294624, -0.336682354469653]
    # theta = [ 0.06781779,  0.44853928, -0.00295598,  0.00440354,  0.25012321,  2.32268618 ,-0.90025305,  1.58695406, -0.04057186, -0.29496854, -2.15145196,  2.31915235 ,-1.83189412, -1.12278177, -0.24124275,  0.12642555, -2.69019782, -0.9835372 , 0.03617707, -0.10909144, -0.29637378, -0.16553028,  0.18251379,  0.39077641 , 0.1489756 ,  1.64906526, -0.22065307, -0.39391793, -0.00763912,  0.10048711 , 0.52361708,  0.40247144,  0.35713522,  0.04261272, -0.05208179,  0.08483703 ,-0.37450726,  0.16595387, -0.07181837,  0.03523731,  0.29876619, -0.44526969 ,-0.34069901]
    
    # After one more pulse:
    # theta = [ 0.06781779,  0.44853928, -0.00295598,  0.00440354,  0.25012321,  2.32268618 ,-0.90025305,  1.58695406, -0.04057186, -0.29496854, -2.15145196,  2.31915235 ,-1.83189412, -1.12278177, -0.24124275,  0.12642555, -2.69019782, -0.9835372 , 0.03617707, -0.10909144, -0.29637378, -0.16553028,  0.18251379,  0.39077641 , 0.1489756 ,  1.64906526, -0.22065307, -0.39391793, -0.00763912,  0.10048711 , 0.52361708,  0.40247144,  0.35713522,  0.04261272, -0.05208179,  0.08483703 ,-0.37450726,  0.16595387, -0.07181837,  0.03523731,  0.29876619, -0.44526969 ,-0.34069901, 0.0, 0.0, 0.0, 0.0, 0.0]

    # from zero:
    # theta = [0.00654194174620981, 0.6636812735981107, 0.021304960744685547, 0.04749089698072337, 0.3092946959253189, 2.149626971833412, -0.8237524695283376, 1.8683519640015558, -0.02354506098362696, -0.32111181548589374, -1.8822250297044363, 2.2940220024100637, -2.3612790897810108, -1.149268387804012, -0.23723333041764716, 0.1834380353097434, -2.7308958942061485, -0.935058540393366, 0.014622301203753423, -0.13665059378679328, -0.37300380063515504, -0.13286792736487762, 0.12335874225550639, 0.3914830532397586, 0.17632788950467979, 1.634479769675761, -0.15130098368657607, -0.39310124217111386, -0.01759570391499231, 0.08601053378508056, 0.6157627108094046, 0.34965992082806974, 0.25779944226565754, 0.043699151159949934, -0.08480605669573209, 0.13588737928612116, -0.41005230688907784, 0.13260762669641163, -0.04244080872688815, 0.062460408188987016, 0.2782901943011341, -0.41430694601825047, -0.38205958551450253, 0.00929391800716569, 0.006783503135437623, 0.005616490940831427, -0.04079164157194315, 0.000463732885749065]
    # theta = [0.0078237314498983, 0.6404821656740626, 0.029838554207664, 0.0477657727996327, 0.3043992240649199, 2.1670507326175628, -0.8207939464659308, 1.8550217120775603, -0.0247079857140462, -0.3226266881258821, -1.8746036593062865, 2.306662569001788, -2.373597735452707, -1.147555191468534, -0.2363091673968706, 0.2133566317133856, -2.784884523088426, -0.9494569156590542, 0.0153259593399963, -0.135238791094732, -0.3823558258870857, -0.07131962930429811, 0.12301824809124241, 0.3926574698177276, 0.1766388892132163, 1.6481649544736787, -0.13153842132098692, -0.402721257022389, -0.01989360346089815, 0.0863771745681564, 0.5829771778174357, 0.3577701724879284, 0.25589679370418195, 0.0450951726512776, -0.08745402944023056, 0.1221457645613145, -0.4151106429120571, 0.1566083317140813, -0.04290269486904774, 0.06106638557853253, 0.29454263977594275, -0.4591092830673435, -0.3921718706674605, 0.012105314685030604, 0.011653984185107089, 0.009399586231536349, -0.028021176705190394, 0.00020663036265098] 
    # theta =  [0.0078237314498983, 0.6437147457772157, 0.061177259226900704, 0.0477426141094454, 0.30450038031576343, 2.1649882630581834, -0.8207939464659308, 1.8532257067532352, -0.024601040457993134, -0.3226266881258821, -1.8746036593062865, 2.306662569001788, -2.373597735452707, -1.1473964106327188, -0.2359748905891994, 0.21766474971976157, -2.7862201687348183, -0.9485669231807865, 0.015482357824596553, -0.135238791094732, -0.3825135104927559, -0.0713196293042981, 0.12385963890029864, 0.3926575695619611, 0.1769548583138194, 1.6483725983407038, -0.1316171295125055, -0.39801912206488144, -0.0198936034608981, 0.0863771745681564, 0.5829771778174357, 0.3586882581226871, 0.26205990202182927, 0.0450951726512776, -0.08672552850554432, 0.12295755876805553, -0.4162525114341828, 0.16275625458538284, -0.04341901262951062, 0.05991457448428747, 0.2989246554722633, -0.4591092830673435, -0.3921718706674605, 0.012160451097526292, 0.011842479932617792, 0.007209090802482065, -0.022618909394532025, 0.0018413237297373935]
    # theta = [-0.026600239120657607, 0.2793564717421055, -0.06672996067144915, -0.008247238680446196, 0.17644190531733783, 0.000729042005309486, -0.00013067980968818052, -0.0018359913360604311, -0.00019879799780479193, 0.001694080066779465, 2.357452986219442, -1.1275664026784735, 1.5467705137367358, -0.04888922332965594, -0.29234333380203414, -2.248712651268007, 2.191997905097832, -1.5616587694498352, -1.1028781530889833, -0.24357828248869431, 0.08078222408424489, -2.622544900850383, -1.059010108851706, 0.024695518032437698, -0.11549665523010694, -0.2863154764623166, -0.20660456389257031, 0.05110699449212361, 0.40465592035605236, 0.16920195784090666, 1.7157157239309728, -0.3819116248903675, -0.2938221538063108, -0.00808229430671383, 0.10284851876580095, 0.5947876163742636, 0.3728779186417284, 0.4167475873749059, 0.04845670189233489, -0.06212247773555266, 0.042177678603930985, -0.22079466614052246, 0.0826710059258389, -0.08497625538372848, 0.047680284158068575, 0.17980139406935952, -0.40890380549434024, -0.3442441414309657, -0.3133859824096269, 0.9923389622053687, 0.0032813940126408207, -0.9500359588079057, 1.093901994454836]
    theta = [
        -0.0271572329451953 , +0.2788367004719841 , -0.0678403374077702 , -0.0082506822771931 , +0.1733765239890844 ,
        +0.0011661990355364 , +0.0007180602538434 , -0.0013498272470005 , -0.0003055606769512 , +0.0047376946777496 ,
        +2.3568775720212400 , -1.1282230247614091 , +1.5469743301886427 , -0.0488894148678248 , -0.2923492659592214 ,
        -2.2486978618398936 , +2.1921141063534053 , -1.5627785844391797 , -1.1029020852920808 , -0.2435793263584259 ,
        +0.0804261930421828 , -2.6226131401256243 , -1.0585465418140427 , +0.0246955180324377 , -0.1154957560262728 ,
        -0.2863583069236421 , -0.2066686192362743 , +0.0509472057639484 , +0.4046559203560524 , +0.1692019578409067 ,
        +1.7157157239309728 , -0.3817480092662203 , -0.2938221538063108 , -0.0080822943067138 , +0.1028335040583288 ,
        +0.5946934800784778 , +0.3728612734462661 , +0.4169058422225110 , +0.0484380062841107 , -0.0621348055047751 ,
        +0.0419372017930186 , -0.2207946661405225 , +0.0827211934821621 , -0.0849865959092378 , +0.0476922547767886 ,
        +0.1800277076712921 , -0.4086975465438076 , -0.3441948700456868 , -0.3133859824096269 , +0.9923516656332504 ,
        +0.0035626060000785 , -0.9504445811738729 , +1.0939276330729646
    ]

    operations  = [
        rotation, p2_pulse, rotation, p2_pulse, rotation, p2_pulse, rotation, p2_pulse, rotation, p2_pulse, rotation, p2_pulse, rotation, p2_pulse, rotation, p2_pulse, rotation,  p2_pulse, rotation, p2_pulse, rotation
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
    
    


    
def optimized_Sx2_pulses_by_partial_repetition(
    num_moments:int=40, 
    max_iter_per_attempt=3*int(1e3),
    max_error_per_attempt=1e-9,
    num_free_params=20,
    sigma=0.0002
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
    results = optimized_Sx2_pulses_by_partial_repetition()
    print("Done.")