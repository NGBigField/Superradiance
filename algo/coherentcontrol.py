# ==================================================================================== #
# |                                 Imports                                          | #
# ==================================================================================== #
if __name__ == "__main__":
    import pathlib, sys
    sys.path.append(str(pathlib.Path(__file__).parent.parent))

# Everyone needs numpy:
import numpy as np
from numpy import pi

# For matrix-operations: 
from scipy.linalg import expm  # matrix exponential
from numpy.linalg import matrix_power  

# For typing hints:
from typing import (
    Any,
    Callable,
    Literal,
    Tuple,
    Optional,
    List,
    ClassVar,
    Final,
    Union,
    Generator,
    overload,
    TypeVar
)

# import our helper modules
from utils import (
    arguments,
    assertions,
    numpy_tools as np_utils,
    visuals,
    strings,
    decorators,
    lists,
    maths
)

# For measuring time:
import time

# For visualizations:
import matplotlib.pyplot as plt  # for plotting test results:
from matplotlib.axes import Axes  # for type hinting:

# For OOP:
from dataclasses import dataclass, field

# For simulating state decay:
from physics.emitted_light.main import decay
from physics.evolution import (
    evolve,
    Params as evolution_params,
)

# For printing progress:
from algo.metrics import purity

# for copying input:
from copy import deepcopy

# For checking on real fock states:
from physics.fock import Fock

# ==================================================================================== #
# |                                  Constants                                       | #
# ==================================================================================== #
OPT_METHOD : Final = 'COBYLA'


# ==================================================================================== #
# |                                 Helper Types                                     | #
# ==================================================================================== #
@dataclass
class _PulseSequenceParams():
    xyz : Tuple[float]
    pause : float

_DensityMatrixType = np.matrix   # type alias

@dataclass
class Operation():
    num_params : int 
    function : Callable[[_DensityMatrixType, List[float] ], _DensityMatrixType]
    string_func : Callable[[List[float]], str] = None
    string : str = None
    positive_params_only : bool = False
    rotation_params : List[int] = None
    name : str = ""
    
    def get_string(self, params:List[float]) -> str:
        if self.string_func is not None:
            return self.string_func(*params)
        elif self.string is not None:
            return self.string
        else:
            return None
        
    def get_outputs(self, in_state:_DensityMatrixType, params:List[float], num_intermediate:int) -> Tuple[
        _DensityMatrixType,
        List[_DensityMatrixType]
    ]:
        op_output = self.function(in_state, *params, num_intermediate=num_intermediate)                
        if isinstance(op_output, list):
            out_state = op_output[-1]
            transition_states = op_output
        elif isinstance(op_output, (np.matrix, np.ndarray) ):
            out_state = op_output
            transition_states = [op_output]                
        else: 
            raise ValueError(f"Bug")
        
        return out_state, transition_states



# ==================================================================================== #
#|                                Inner Functions                                     |#
# ==================================================================================== #

def _assert_N(N:int) -> None:
    assertions.even(N) # N must be even    

def _J(N:int) -> int : 
    _assert_N(N)
    return N//2

def _M(m:int, J:int) :
    return m-J

def _mat_size(N: int) -> Tuple[int, int]:
    return (N+1, N+1)

def _d_plus_minus(M:int, J:int, pm:int) -> float:
    """ _summary_

    N: even integer
    M: index from 0 to N (including. i.e. N+1 possible values)
    pm : +1 or -1

    return d plust\minus according to the relation:
    D^{\pm}\ket{J,M}=d^{\pm}\ket{J,M}==\sqrt{J(J+1)-M(M\pm1)}\ket{J,M\pm1}
    """
    d = np.sqrt(
        J*(J+1)-M*(M+pm)
    )
    return d

def _D_plus_minus_mats(N:int) -> Tuple[np.matrix, np.matrix] :
    # Derive sizes:
    mat_size = _mat_size(N)
    J = _J(N)
    # Init outputs:    
    D = {}
    D[+1] = np.zeros(mat_size)
    D[-1] = np.zeros(mat_size)
    # Iterate over diagonals:
    for m in range(mat_size[0]):
        M = _M(m, J)
        i = m

        for pm in [-1, +1]:
            # derive matrix indices:
            j = m + pm
            if j>=N+1 or j<0:
                continue
            # compute and assign to matrix
            d = _d_plus_minus(M, J, pm)
            D[pm][i,j] = d

    return D[+1], D[-1]

def _Sz_mat(N:int) -> np.matrix :
    mat_size = _mat_size(N)
    J = _J(N)
    Sz = np.zeros(mat_size)
    for m in range(mat_size[0]):
        M = _M(m, J)
        Sz[m,m] = M
    return Sz

def _float_to_str_except_zeros(x:float|int, num_decimals:int|None=None)->str:
    if num_decimals is None:
        return f"{num_decimals}"
    if x==0:
        return f"0"
    return f"{x:.{num_decimals}f}"

def _deal_costum_params(
    operations: List[Operation],
    theta: Optional[Union[List[float], np.ndarray]]=None
) -> List[List[float]]:
    # Check lengths:
    total_num_params = sum([op.num_params for op in operations])
    if isinstance(theta, np.ndarray):
        # Assert 1D array:
        assert theta.ndim==1
        theta = theta.tolist()
        # Assert correct total length
    elif theta is None and total_num_params==0:
        theta = np.array([])
    assert len(theta)==total_num_params
    # init:
    all_ops_params : list = []
    # iterate:
    i = 0
    for operation in operations:
        num_params = operation.num_params
        op_params = theta[i:(i+num_params)]
        all_ops_params.append(op_params)
        i = i + num_params
    return all_ops_params
        
    
        
def _deal_params(theta: Union[List[float], np.ndarray]) -> List[_PulseSequenceParams] :
    # Constants:
    NUM_PULSE_PARAMS : Final = 4
    # Check length:
    if isinstance(theta, np.ndarray):
        # Assert 1D array:
        assert theta.ndim==1
        theta = theta.tolist()
    assert isinstance(theta, list), f"Input `theta` must be of type `list`"
    l = len(theta)
    p = (l+1)/NUM_PULSE_PARAMS
    assertions.integer(p, reason=f"Length of `theta` must be an integer 4*p - 1 where p is the number of pulses")

    # prepare output and iteration helpers:
    result : List[_PulseSequenceParams] = []
    counter : int = 0
    crnt_pulse_params_list : List[float] = []

    # deal:
    for param in theta:
        # Check input:
        assert isinstance(param, (int, float)), f"Input `theta` must be a list of floats"
        # Add to current pulse:
        counter += 1
        crnt_pulse_params_list.append( param )
        # Check if to wrap-up:
        if counter >= NUM_PULSE_PARAMS:
            # append results:
            result.append(
                _PulseSequenceParams(
                    xyz = crnt_pulse_params_list[0:3],
                    pause = crnt_pulse_params_list[3]
                )
            )
            # reset iteration helpers:
            counter = 0
            crnt_pulse_params_list = []

    # Last iteration didn't complete to 4 since the last pulse doesn't require a pause:
    result.append(
        _PulseSequenceParams(
            xyz = crnt_pulse_params_list[0:3],
            pause = 0
        )
    )

    # end:
    return result
    
def _num_divisions_from_num_intermediate_states(num_intermediate_states:int) -> int:
    num_intermediate_states_error_msg = f"`num_intermediate_states` must be a non-negative number."
    num_intermediate_states = assertions.integer(num_intermediate_states)
    if num_intermediate_states > 0:
        num_divides = num_intermediate_states
    elif num_intermediate_states == 0:
        num_divides = 1
    else:
        raise ValueError(num_intermediate_states_error_msg)
    return num_divides

def _list_of_intermediate_pulsed_states(state:_DensityMatrixType, p:np.matrix, num_divides:int, num_intermediate_states:int) -> List[_DensityMatrixType]:
    # Hermitian conj of pulse:
    pH = p.getH()
    # prepare outputs:
    states : List[_DensityMatrixType] = []
    crnt_state = deepcopy(state)
    if num_intermediate_states>0:
        states.append(crnt_state)   # add initial state
    # Apply pulse:
    for time_step in range(num_divides):            
        crnt_state = p * crnt_state * pH
        states.append(crnt_state)
    return states

# ==================================================================================== #
# |                            Declared Functions                                    | #
# ==================================================================================== #




def S_mats(N:int) -> Tuple[ np.matrix, np.matrix, np.matrix ] :
    # Check input:
    _assert_N(N)
    # Prepare Base Matrices:
    D_plus, D_minus = _D_plus_minus_mats(N)
    # Derive X, Y, Z Matrices
    Sx = ( D_plus + D_minus ) * (1/2)
    Sy = ( -1j*D_plus + 1j*D_minus ) * (1/2)
    Sz = _Sz_mat(N)
    # Return:
    return Sx, Sy, Sz

def _stark_shift_mat(mat_size:int, indices:List[int], shifts:List[float]) -> np.matrix:
    mat =  np.matrix( np.zeros(shape=(mat_size, mat_size), dtype=np.complex64) )
    for i in range(mat_size):
        mat[i, i] = 1
    for i, shift in zip(indices, shifts):
        mat[i, i] = np.exp( 1j * shift )
    return mat

def pulse(
    x  : float, 
    y  : float, 
    z  : float, 
    Sx : np.matrix, 
    Sy : np.matrix, 
    Sz : np.matrix,
    c  : float = 1.0  # Scaling param
) -> np.matrix :
    exponent = 1j*(x*Sx + y*Sy + z*Sz)*c
    return np.matrix( expm(exponent) )  # e^exponent


# ==================================================================================== #
# |                                 Classes                                          | #
# ==================================================================================== #

class SPulses():
    def __init__(self, N:int) -> None:
        self.N = N
        Sx, Sy, Sz = S_mats(N)
        D_plus, D_minus = _D_plus_minus_mats(N)
        self.Sx = Sx 
        self.Sy = Sy 
        self.Sz = Sz
        self.Sp = D_plus  + 0*1j  # force casting to complex
        self.Sm = D_minus + 0*1j  # force casting to complex

    def __repr__(self) -> str:
        res = f"S Pulses with N={self.N}: \n"
        for (mat, name) in [
            (self.Sx, 'Sx'),
            (self.Sy, 'Sy'),
            (self.Sz, 'Sz')
        ]:
            res += f"{name}:\n"
            res += np_utils.mat_str(mat)+"\n"
        return res

class SequenceMovieRecorder():
    
    ViewingAngles = visuals.ViewingAngles
    
    @staticmethod
    def _default_score_str_func(state:_DensityMatrixType) -> str:
        return ""

    @dataclass
    class Config():
        active : bool = False
        show_now : bool = True
        fps : int = 5
        num_transition_frames : int|tuple[int, int, int] = 10
        num_freeze_frames : int = 5
        horizontal_figure : bool = True
        bloch_sphere_config : visuals.BlochSphereConfig = field( default_factory=visuals.BlochSphereConfig )
        score_str_func : Optional[Callable[[_DensityMatrixType], str]] = None
        temp_dir_name : str = ""
        final_frame_duration_multiplier : int = 3

        def get_num_transition_frames_based_on_operation(self, params:list[float], operation:Operation)->int:
            if not self.active:
                return 1
            elif isinstance(self.num_transition_frames, int):
                return self.num_transition_frames 
            elif isinstance(self.num_transition_frames, tuple):
                if operation.name == "rotation":
                    return self.num_transition_frames[0]
                elif operation.name == "squeezing":
                    l2 = np.sqrt(sum([val**2 for val in params]))
                    _range = (self.num_transition_frames[1], self.num_transition_frames[2])
                    return _squeezing_num_transition_based_on_strength(strength=l2, requested_range=_range)
            
            raise TypeError("Not an expected type") 

    
    def __init__(
        self, 
        initial_state : Optional[_DensityMatrixType] = None,
        config : Optional[Config] = None,
    ) -> None:
        # Base properties:        
        self.config : SequenceMovieRecorder.Config = arguments.default_value(config, default_factory=SequenceMovieRecorder.Config )
        if self.config.active:
            self.figure_object : visuals.MatterStatePlot = visuals.MatterStatePlot(
                initial_state=initial_state,
                bloch_sphere_config=self.config.bloch_sphere_config,
                horizontal=self.config.horizontal_figure
            )            
            self.video_recorder : visuals.VideoRecorder = visuals.VideoRecorder(fps=self.config.fps, temp_dir_name=self.config.temp_dir_name)
        else:
            self.figure_object = None
            self.video_recorder = None
        # Score strung func:
        if self.config.score_str_func is None:
            self.config.score_str_func = SequenceMovieRecorder._default_score_str_func
        self.score_str_func : Callable[[np.matrix], str] = self.config.score_str_func
        # Keep last state:
        self.last_state : _DensityMatrixType = initial_state
        
    def _record_single_state(
        self,
        state : _DensityMatrixType,
        duration : int,  # number of repetitions of the same frame:
        title : Optional[str]=None
    ) -> None:
        score_str = self._derive_score_str(state)
        fontsize = 16 if self.config.horizontal_figure else 12
        self.figure_object.update(state, title=title, show_now=self.config.show_now, score_str=score_str, fontsize=fontsize)
        self.video_recorder.capture(fig=self.figure_object.figure, duration=duration)

    def _derive_score_str(self, state:_DensityMatrixType) -> Union[str, None]:
        if self.score_str_func is None:
            return None
        else:
            return self.score_str_func(state)

    def final_state(self)->None:
        if not self.is_active:
            return  # We don't want to record a video
        length_multiplier = self.config.final_frame_duration_multiplier
        default_duration = self.video_recorder.frames_duration[-1]
        new_duration = int(round(default_duration*length_multiplier))
        self.video_recorder.frames_duration[-1] = new_duration

    def record_transition(
        self, 
        transition_states:List[_DensityMatrixType], 
        title:str, 
    ) -> None:
        # Check inputs:
        if not self.is_active:
            return  # We don't want to record a video
        final_state = transition_states[-1]
        
        # Capture shots: (transition and freezed state)
        prog_bar = strings.ProgressBar(len(transition_states), print_prefix="Capturing transition frames... ")
        for is_first, is_last, transition_state in lists.iterate_with_edge_indicators(transition_states):
            prog_bar.next()
            if np.array_equal(transition_state, self.last_state):
                continue
            if is_last:
                assert np.array_equal(transition_state, final_state)
                self._record_single_state(final_state, title=title, duration=self.config.num_freeze_frames)
            else:
                self._record_single_state(transition_state, title=title, duration=1 )
        prog_bar.clear()
        
        # Keep info for next call:
        self.last_state = deepcopy(final_state)
        
    def write_video(self) -> None:
        if not self.is_active:
            return
        self.video_recorder.write_video()
        self._close()
        
    def _close(self) -> None:
        self.figure_object.close()

    @property
    def is_active(self) -> bool:
        return self.config.active

    
class CoherentControl():   

    # Class Attributes:
    _default_state_decay_resolution : ClassVar[int] = 1000
    MovieConfig = SequenceMovieRecorder.Config
    

    # ==================================================== #
    #|                    constructor                     |#
    # ==================================================== #

    def __init__(
        self, 
        num_atoms:int,
        gamma:float=1.0,
    ) -> None:
        # Keep basic properties:        
        self._num_moments = num_atoms
        self.gamma = gamma
        # define basic pulses:
        self.s_pulses = SPulses(num_atoms)

    # ==================================================== #
    #|                  static methods                    |#
    # ==================================================== #        
    @staticmethod
    def num_params_for_pulse_sequence(num_pulses:int) -> int:
        assertions.integer(num_pulses)
        return num_pulses*4-1

    # ==================================================== #
    #|                   inner functions                  |#
    # ==================================================== #

    def stark_shift_and_rot_mat(
        self,
        mat_size:int, 
        global_strength:float,
        stark_shift_indices:List[int],
        stark_shift_strength:float,
        rotation_indices:List[int],
        rotation_angle:float,
    ) -> np.matrix:

        mat =  np.matrix( np.zeros(shape=(mat_size, mat_size), dtype=np.complex64) )
        for i in stark_shift_indices:
            mat[i, i] = stark_shift_strength 

        if 0 in rotation_indices:
            mat += self.s_pulses.Sx * np.cos(rotation_angle)
        if 1 in rotation_indices:
            mat += self.s_pulses.Sy * np.sin(rotation_angle)
        if 2 in rotation_indices:
            mat += self.s_pulses.Sz

        res = expm(1j * mat * global_strength)

        return np.matrix( res )

    def _pulse(self, x:float=0.0, y:float=0.0, z:float=0.0, power:int=1) -> np.matrix:
        # Check inputs:
        reason = "`power` must be a positive integer"
        assertions.integer(power, reason=reason)
        assert power>0, reason
        # Derive Operators (with powers, if needed)        
        Sx = matrix_power(self.s_pulses.Sx, power) 
        Sy = matrix_power(self.s_pulses.Sy, power) 
        Sz = matrix_power(self.s_pulses.Sz, power)
        return pulse(x,y,z, Sx,Sy,Sz)

    def _state_decay_old_iterative_method(
        self,
        state:np.matrix, 
        time:float, 
        time_steps:Optional[int]=None,
    )->np.matrix:
        # Complete missing inputs:
        time_steps = arguments.default_value(time_steps, self._default_state_decay_resolution)    
        assertions.integer(time_steps)
        # Params:
        params = evolution_params(
            N = self.num_moments,
            dt = time/time_steps,
            Gamma = self.gamma
        )
        # Compute:
        crnt_state = deepcopy(state) 
        for _ in range(time_steps):
            crnt_state = evolve( rho=crnt_state, params=params )
        # Return:
        return crnt_state

    # ==================================================== #
    #|                 declared functions                 |#
    # ==================================================== #

    class StandardOperations():

        def __init__(self, coherent_control:"CoherentControl", num_intermediate_states:int=0) -> None:
            self.num_intermediate_states = num_intermediate_states
            self.coherent_control : CoherentControl = coherent_control


        def _deal_values_to_indices(self, values:List[float], indices:List[int]) -> List[float]:
            out = [0, 0, 0]
            for ind, val in zip(indices, values):
                out[ind] = val
            return out

        def _power_pulse_string_func(self, theta:List[float], indices:List[int], power:int, num_decimals:int=6):
            values = self._deal_values_to_indices(values=theta, indices=indices)
            _s = lambda x: _float_to_str_except_zeros(x, num_decimals)
            return f"Power-{power} pulse: [{_s(values[0])}, {_s(values[1])}, {_s(values[2])}]"

        def _power_pulse_func(self, rho:_DensityMatrixType, theta:List[float], indices:List[float], power:int, num_intermediate:int|None=None) -> _DensityMatrixType:
            values = self._deal_values_to_indices(values=theta, indices=indices)

            if num_intermediate is None:
                num_intermediate = self.num_intermediate

            return self.coherent_control.pulse_on_state_with_intermediate_states(
                rho, x=values[0], y=values[1], z=values[2], power=power, num_intermediate=num_intermediate
            )
            
        def power_pulse(self, power:int) -> Operation:
            indices = [0, 1, 2]  # all indices
            return self.power_pulse_on_specific_directions(power=power, indices=indices)

        def power_pulse_on_specific_directions(self, power:int, indices:List[int] = [0,1,2]) -> Operation:
            def _function(rho, *theta, num_intermediate:int|None=None): 
                return self._power_pulse_func(rho, power=power, theta=theta, indices=indices, num_intermediate=num_intermediate)
            
            op = Operation(
                num_params = len(indices),
                function = _function,
                string_func = lambda *theta: self._power_pulse_string_func(theta=theta, indices=indices, power=power )
            )
            if power==1:
                op.name = "rotation"
            elif power==2:
                op.name = "squeezing"
            else:
                op.name = f"power-{power} pulse"                
            return op
        
        def squeezing(self, axis:Optional[Tuple[float, float]]=None) -> Operation:
            # Define num params:
            if axis is None:
                num_params = 3
            else:
                num_params = 1
            # Define functions:
            def _deal_inputs(*theta) -> Tuple[ float, Tuple[float, float]]:
                strength = theta[0]
                if axis is None:
                    assert len(theta)==3
                    axis_input = (theta[1], theta[2])
                else:
                    assert len(theta)==1
                    axis_input = (1, 0) 
                return strength, axis_input
            def _func(rho, *theta):
                strength, axis_input = _deal_inputs(*theta)
                return self.coherent_control.squeezing_with_intermediate_states(
                    rho, strength=strength, axis=axis_input, num_intermediate_states=self.num_intermediate_states
                )
            def _str_func(*theta):
                strength, axis_input = _deal_inputs(*theta)
                return f"squeezing on direction {axis_input} with strength {strength}"

            return Operation(
                num_params=num_params,
                function=_func,
                string_func=_str_func
            )

        def stark_shift(self, indices:Optional[List[int]]=None) -> Operation:
            if indices is None:
                num_params = self.density_matrix_size
            else:
                num_params = len(indices)
            return Operation(
                num_params = num_params,
                function = lambda rho, *theta: self.coherent_control.stark_shift_with_intermediate_states(
                    rho, num_intermediate_states=self.num_intermediate_states, indices=indices, shifts=theta
                ),
                string_func = lambda *theta: f"Stark-shift on indices {indices} with values {theta}"
            )

        def stark_shift_and_rot(self, stark_shift_indices:List[int]=[1], rotation_indices:List[int] = [0, 1] ):
            # Check inputs:
            num_stark_shifts = len(stark_shift_indices)
            assert num_stark_shifts<2, "We don't yet support many stark-shifts"

            # prepare inputs:
            if len(rotation_indices) in [0, 1]:
                num_rotation_directions = 0
            else:
                num_rotation_directions = 1

            num_params = num_stark_shifts + num_rotation_directions + 1
            
            def _deal_params(*theta)->Tuple[
                float,  # global_strength
                float,  # rotation_angle
                float   # stark_shift_strength
            ]:
                global_strength = theta[0]

                if num_rotation_directions == 0:
                    rotation_angle = 0.0

                    if num_stark_shifts == 0:
                        stark_shift_strength = 0
                    elif num_stark_shifts == 1:
                        stark_shift_strength = theta[1]
                    else:
                        raise NotImplementedError(f"We don't yet support many stark-shifts")

                elif num_rotation_directions == 1:
                    rotation_angle = theta[1]

                    if num_stark_shifts == 0:
                        stark_shift_strength = 0
                    elif num_stark_shifts == 1:
                        stark_shift_strength = theta[2]
                    else:
                        raise NotImplementedError(f"We don't yet support many stark-shifts")

                else:
                    raise NotImplementedError(f"We don't yet support many thetas")                
                
                return global_strength, rotation_angle, stark_shift_strength

            def _func(rho:_DensityMatrixType, *theta)->List[_DensityMatrixType]:
                # Deal params:
                global_strength, rotation_angle, stark_shift_strength = _deal_params(*theta)

                # Call func:
                return self.coherent_control.stark_shift_and_rot_with_intermediate_states(
                    state = rho, 
                    global_strength = global_strength,
                    num_intermediate_states = self.num_intermediate_states, 
                    stark_shift_indices = stark_shift_indices,
                    rotation_indices = rotation_indices,
                    rotation_angle = rotation_angle,
                    stark_shift_strength = stark_shift_strength
                )
                
            def _str_func(*theta) -> str:
                # Deal params:
                global_strength, rotation_angle, stark_shift_strength = _deal_params(*theta)
                return f"Stark-shift on {stark_shift_indices} with global_strength={global_strength} and stark_shift_strength={stark_shift_strength} \n"+\
                    f"and rotation on {rotation_indices} with rotation_angle={rotation_angle}"

            # Return operation
            return Operation(
                num_params = num_params,
                function = _func,
                string_func = _str_func,
                rotation_params=[]
            )

        def decay(self, time_steps_resolution:int=10001) -> Operation:
            return Operation(
                num_params = 1,
                function = lambda rho, t: self.coherent_control.state_decay_with_intermediate_states(
                    rho, time=t, num_intermediate_states=self.num_intermediate_states, time_steps_resolution=time_steps_resolution
                ),
                string_func = lambda t: f"Decay with time={t}",
                positive_params_only=True
            )

    def standard_operations(self, num_intermediate_states:int=0) -> StandardOperations:
        return CoherentControl.StandardOperations(self, num_intermediate_states=num_intermediate_states)

    def stark_shift_and_rot_with_intermediate_states(
        self, 
        state:_DensityMatrixType, 
        global_strength:float,
        num_intermediate_states:int = 0, 
        stark_shift_indices:List[int] = [1],
        rotation_indices:List[int] = [0, 1],
        rotation_angle:float=1.0,
        stark_shift_strength:float=1.0
    ) -> List[_DensityMatrixType]:
        # Check input:
        matrix_size = state.shape[0]
        assert state.shape[0]==state.shape[1]
        # Create fractional pulse strength
        num_divides = _num_divisions_from_num_intermediate_states(num_intermediate_states)
        strength_frac = global_strength/num_divides
        # Create Matrix:
        p = self.stark_shift_and_rot_mat(
            mat_size=matrix_size, global_strength=strength_frac, stark_shift_indices=stark_shift_indices, 
            rotation_indices=rotation_indices, rotation_angle=rotation_angle, stark_shift_strength=stark_shift_strength
        )
        return _list_of_intermediate_pulsed_states(state=state, p=p, num_divides=num_divides, num_intermediate_states=num_intermediate_states)

    def stark_shift_with_intermediate_states(
        self, 
        state:_DensityMatrixType, 
        shifts:List[float],
        num_intermediate_states:int=0, 
        indices:Optional[List[int]]=None,  # Defaults to all matrix indices
    ) -> List[_DensityMatrixType]:
        # Check input:
        matrix_size = state.shape[0]
        indices = args.default_value(indices, list(range(matrix_size)))
        assert len(indices)==len(shifts), "Lists `indices` and `shifts` must be of the same length."
        assert state.shape[0]==state.shape[1]
        # Create fractional pulse strength
        num_divides = _num_divisions_from_num_intermediate_states(num_intermediate_states)
        frac_shifts = [shift/num_divides for shift in shifts]
        # Create Matrix:
        p = _stark_shift_mat(matrix_size, indices=indices, shifts=frac_shifts)
        return _list_of_intermediate_pulsed_states(state=state, p=p, num_divides=num_divides, num_intermediate_states=num_intermediate_states)

    def pulse_on_state(self, state:_DensityMatrixType, x:float=0.0, y:float=0.0, z:float=0.0, power:int=1) -> _DensityMatrixType: 
        return self.pulse_on_state_with_intermediate_states(state=state, num_intermediate=0, x=x, y=y, z=z, power=power)[-1]

    def pulse_on_state_with_intermediate_states(self, state:_DensityMatrixType, num_intermediate:int=0, x:float=0.0, y:float=0.0, z:float=0.0, power:int=1) -> List[_DensityMatrixType]: 
        # Check input:
        num_divides = _num_divisions_from_num_intermediate_states(num_intermediate)
        # Divide requested pulse into fragments
        frac_x = x / num_divides
        frac_y = y / num_divides
        frac_z = z / num_divides
        p = self._pulse(frac_x, frac_y, frac_z, power=power)
        return _list_of_intermediate_pulsed_states(state=state, p=p, num_divides=num_divides, num_intermediate_states=num_intermediate)  

    def state_decay(self, state:_DensityMatrixType, time:float, time_steps_resolution:int=10001) -> _DensityMatrixType: 
        return self.state_decay_with_intermediate_states(state=state, time=time, num_intermediate_states=0, time_steps_resolution=time_steps_resolution)[-1]        

    def state_decay_with_intermediate_states(
        self, 
        state:_DensityMatrixType, 
        time:float, 
        num_intermediate_states:int=0,
        time_steps_resolution:int=10001
    ) -> List[_DensityMatrixType] :
        # Check inputs:
        assertions.density_matrix(state, robust_check=True)  # allow matrices to be non-PSD or non-Hermitian
        num_intermediate_states = assertions.integer(num_intermediate_states, reason=f"`num_intermediate_states` must be a non-negative integer")
        if time==0:
            return [state]
        assert time>0, f"decay time must be a positive number. got {time}"
        # Complete missing inputs:
        assertions.integer(time_steps_resolution)            
        # Convert matrix to ndarray:
        if isinstance(state, np.matrix):
            state = np.array(state)
        # Compute:
        rho_at_all_times = decay(
            rho=state,  
            delta_t=time,
            gamma=self.gamma,
            num_time_steps=time_steps_resolution
        )
        # Return results in requested indices:
        num_times = rho_at_all_times.shape[2]
        if num_intermediate_states>0:
            indices = np.floor( np.linspace(0, num_times-1, num_intermediate_states+1) ).astype(int)
        elif num_intermediate_states==0:
            indices = [num_times-1]
        else:
            raise ValueError(f"`num_intermediate_states` must be a non-negative integer")
        return [rho_at_all_times[:,:,ind] for ind in indices ]

    def _z_squeezing_operator(self, strength:float) -> np.matrix:        
        # Bring matrices
        Sz = self.s_pulses.Sz   # force casting to complex
        # Define params:
        w_0 = 0.495
        w_j = 0.010
        num_atoms = self.num_moments
        # Create operator:
        exponent =  - 1j * ( Sz * ( w_0 / 2 ) -  (Sz@Sz) * (w_j / num_atoms) ) * strength 
        op = expm(exponent)
        return np.matrix( op )

    def _squeezing_operator(self, strength:float, axis:Tuple[float, float]) -> np.matrix:
        # Bring matrices
        Sp = self.s_pulses.Sp   # force casting to complex
        Sm = self.s_pulses.Sm   # force casting to complex
        # Define axis:
        xi = axis[0] + 1j*axis[1]
        # create operator:
        # exponent = ( Sp@Sp * xi + Sm@Sm * np.conj(xi) ) * strength
        # op = expm(-1j*exponent)
        exponent = ( Sp@Sp * np.conj(xi) - Sm@Sm * xi ) * strength
        op = expm(exponent)
        return np.matrix( op )

    def squeezing(self, state:_DensityMatrixType, strength:float, axis:Tuple[float, float]=(1,0)) -> _DensityMatrixType:
        return self.squeezing_with_intermediate_states(state=state, strength=strength, num_intermediate_states=0, axis=axis)[-1]        

    def squeezing_with_intermediate_states(self, state:_DensityMatrixType, strength:float, num_intermediate_states:int=0, axis:Tuple[float, float]=(1,0)) -> List[_DensityMatrixType] :
        # Check input:
        num_divides = _num_divisions_from_num_intermediate_states(num_intermediate_states)
        axis = arguments.default_value(axis, (1,0))
        # Divide requested pulse into fragments
        strength_frac = strength/num_divides
        p = self._squeezing_operator(strength=strength_frac, axis=axis)
        # Return output:
        return _list_of_intermediate_pulsed_states(state=state, p=p, num_divides=num_divides, num_intermediate_states=num_intermediate_states)

    def z_squeezing(self, state:_DensityMatrixType, strength:float) -> List[_DensityMatrixType] :
        return self.z_squeezing_with_intermediate_states(state=state, strength=strength, num_intermediate_states=0)[-1]

    def z_squeezing_with_intermediate_states(self, state:_DensityMatrixType, strength:float, num_intermediate_states:int=0 ) -> List[_DensityMatrixType] :
        # Check input:
        num_divides = _num_divisions_from_num_intermediate_states(num_intermediate_states)    
        # Divide requested pulse into fragments
        strength_frac = strength/num_divides
        p = self._z_squeezing_operator(strength=strength_frac)
        # Return output:
        return _list_of_intermediate_pulsed_states(state=state, p=p, num_divides=num_divides, num_intermediate_states=num_intermediate_states)

    def custom_sequence(
        self, 
        state:np.matrix, 
        theta: Union[ List[float], np.ndarray, None ],
        operations: List[Operation],
        movie_config : Optional[MovieConfig] = None
    ) -> _DensityMatrixType :
        
        # Check and prepare inputs:
        assertions.density_matrix(state, robust_check=True)
        all_params = _deal_costum_params(operations, theta)
        crnt_state = deepcopy(state)
        movie_config = arguments.default_value(movie_config, default_factory=CoherentControl.MovieConfig)
        assert isinstance(movie_config, CoherentControl.MovieConfig)

        # For sequence recording:
        sequence_recorder = SequenceMovieRecorder(config=movie_config)

        # iterate:
        num_iter = len(operations)
        if movie_config.active: 
            prog_bar = strings.ProgressBar(num_iter, print_prefix="Performing custom sequence...  ")
        else:                   
            prog_bar = strings.ProgressBar.inactive()

        for i, (params, operation) in enumerate(zip(all_params, operations)):    
            prog_bar.next()
            # Check params:
            assert operation.num_params == len(params)
            # Apply operation:
            num_intermediate = movie_config.get_num_transition_frames_based_on_operation(params, operation)
            crnt_state, transition_states = operation.get_outputs(crnt_state, params, num_intermediate)
            # Get title:
            title = operation.get_string(params)
            # Record:
            sequence_recorder.record_transition(transition_states, title=title)
        prog_bar.clear()
        
        sequence_recorder.final_state()  # Just makes the last frame twice as long
        sequence_recorder.write_video()

        # End:
        return crnt_state
    
    def coherent_sequence(
        self, 
        state:np.matrix, 
        theta: Union[ List[float], np.ndarray ],
        record_movie : bool = False,
        movie_config : MovieConfig = MovieConfig()
    ) -> _DensityMatrixType :
        """coherent_sequence Apply sequence of coherent pulses separated by state decay.

        The length of the `theta` is 4*p - 1 
        where `p` is the the number of pulses
        3*p x,y,z parameters for p pulses.
        p-1 decay-time parameters between each pulse.

        Args:
            state (np.matrix): initial density-matrix
            theta (List[float]): parameters.
            record_movie bool: Should the sequence be recorded?
            movie_config MovieConfig: If the sequence should be recorded, these are the configurations.

        Returns:
            np.matrix: final density-matrix
        """
        # Check and prepare inputs:
        assertions.density_matrix(state, robust_check=True)
        params = _deal_params(theta)
        crnt_state = deepcopy(state)

        # For sequence recording:
        movie_config.active = record_movie | movie_config.active
        sequence_recorder = SequenceMovieRecorder(initial_state=crnt_state, config=movie_config)
        num2str = lambda x : strings.formatted(x, width=5, precision=5)
        if sequence_recorder.is_active:
            num_intermediate_states = sequence_recorder.config.num_transition_frames
        else:
            num_intermediate_states = 0

        # iterate:
        for pulse_params in params:
            # Unpack parans:
            x = pulse_params.xyz[0]
            y = pulse_params.xyz[1]
            z = pulse_params.xyz[2]
            t = pulse_params.pause
            # Apply pulse and delay:
            if x != 0 or y != 0 or z != 0:
                transition_states = self.pulse_on_state_with_intermediate_states(state=crnt_state, x=x, y=y, z=z, num_intermediate=num_intermediate_states)
                sequence_recorder.record_transition(transition_states, f"Pulse = [{num2str(x)}, {num2str(y)}, {num2str(z)}]")
                crnt_state = transition_states[-1]
            if t != 0:
                transition_states = self.state_decay_with_intermediate_states(state=crnt_state, time=t, num_intermediate_states=num_intermediate_states)
                sequence_recorder.record_transition(transition_states, f"Decay-Time = {num2str(t)}")
                crnt_state = transition_states[-1]
        
        sequence_recorder.write_video()

        # End:
        return crnt_state
        
    # ==================================================== #
    #|                  setters\getters                   |#
    # ==================================================== #        

    @property
    def num_moments(self) -> int:
        return self._num_moments
    @num_moments.setter
    def num_moments(self, val:Any) -> None:
        self._num_moments = val
        self.s_pulses = SPulses(val)

    @property
    def density_matrix_size(self) -> int:
        return self.num_moments + 1


# ==================================================================================== #
# |                           more inner functions                                   | #
# ==================================================================================== #
def _squeezing_num_transition_based_on_strength(strength:float, requested_range:tuple[int,int])->int:
    linear_interpolation = maths.linear_interpolation_by_range(x=strength, x_range=(0.0, 2*pi), y_range=requested_range)    
    return int(linear_interpolation)


# ==================================================================================== #
# |                                   main                                           | #
# ==================================================================================== #

def _test_record_sequence():
    # Const:
    num_moments:int=4
    num_pulses:int=3
    movie_config=CoherentControl.MovieConfig(
        show_now=True,
        num_transition_frames=5,
        num_freeze_frames=10,
        fps=5,
        bloch_sphere_resolution=30
    )
    # init params:
    num_params = CoherentControl.num_params_for_pulse_sequence(num_pulses=num_pulses)
    theta = np.random.random((num_params))
    initial_state = Fock.ground_state_density_matrix(num_moments)
    # Apply:
    coherent_control = CoherentControl(num_atoms=num_moments)
    final_state = coherent_control.coherent_sequence(state=initial_state, theta=theta, record_movie=True, movie_config=movie_config )
    print("Movie is ready in folder 'video' ")


def _test_pulse_in_steps():
    # Define params:
    num_moments:int=4   
    num_steps:int=20
    fps:int=5
    # Init state:
    initial_state = Fock(0).to_density_matrix(num_moments=num_moments)
    coherent_control = CoherentControl(num_atoms=num_moments)
    # Apply pulse:
    all_pulse_states = coherent_control.pulse_on_state_with_intermediate_states(state=initial_state, num_intermediate=num_steps, x=np.pi )
    # Apply decay time:
    all_decay_states = coherent_control.state_decay_with_intermediate_states(state=all_pulse_states[-1], num_intermediate_states=num_steps, time=0.5)
    # Movie:
    visuals.draw_now()
    state_plot = visuals.MatterStatePlot(block_sphere_resolution=100, initial_state=initial_state)
    video_recorder = visuals.VideoRecorder(fps=fps)
    video_recorder.capture(state_plot.figure, duration=fps)
    def capture(state, duration:int=1):
        state_plot.update(state)
        video_recorder.capture(state_plot.figure, duration=duration)
    for state in all_pulse_states:
        capture(state)
    capture(state, duration=fps)
    for state in all_decay_states:
        capture(state)
    capture(state, duration=fps)
    video_recorder.write_video()

def _test_power_pulse():
    # Define params:
    draw_now:bool=True
    num_moments:int=40
    num_steps1:int=5
    num_steps2:int=20
    fps:int=5
    block_sphere_resolution:int=100
    # Init state:
    initial_state = Fock(0).to_density_matrix(num_moments=num_moments)
    coherent_control = CoherentControl(num_atoms=num_moments)
    # Apply pulse:
    pi_half_transition = coherent_control.pulse_on_state_with_intermediate_states(state=initial_state, num_intermediate=num_steps1, x=np.pi/2, power=1 )
    sz2_transition     = coherent_control.pulse_on_state_with_intermediate_states(state=pi_half_transition[-1], num_intermediate=num_steps2, z=np.pi/8, power=2 )
    # Prepare Movie:
    state_plot = visuals.MatterStatePlot(block_sphere_resolution=block_sphere_resolution, initial_state=initial_state)
    video_recorder = visuals.VideoRecorder(fps=fps)
    video_recorder.capture(state_plot.figure, duration=fps)
    # Helper function:
    def capture(state, duration:int=1):
        state_plot.update(state)
        video_recorder.capture(state_plot.figure, duration=duration)
        if draw_now:
            visuals.draw_now()
    # Capture Movie:
    for state in pi_half_transition:
        capture(state)
    for state in sz2_transition:
        capture(state)
    capture(state, duration=fps)
    video_recorder.write_video()   

def _test_goal_gkp():
    # Import:
    from gkp import goal_gkp_state
    # Define params:
    num_moments:int=40
    block_sphere_resolution:int=200
    # Init state:
    coherent_control = CoherentControl(num_atoms=num_moments)
    gkp = goal_gkp_state(num_moments=num_moments)
    # Plot:
    visuals.plot_city(gkp)
    visuals.draw_now()
    visuals.plot_wigner_bloch_sphere(gkp, num_points=block_sphere_resolution)
    print("Done")
    
def _show_analitical_gkp():
    # Const:
    num_moments:int=20
    num_transition_frames=0 #20
    active_movie_recorder:bool=False
    fps=10

    # Movie config:
    movie_config=CoherentControl.MovieConfig(
        active=active_movie_recorder,
        show_now=False,
        num_freeze_frames=fps,
        fps=fps,
        bloch_sphere_resolution=200,
        # score_str_func=_score_str_func
    )

    ## Define operations:
    coherent_control = CoherentControl(num_atoms=num_moments)
    standard_operations : CoherentControl.StandardOperations = coherent_control.standard_operations(num_intermediate_states=num_transition_frames)
    initial_state = Fock.ground_state_density_matrix(num_moments)

    if num_moments==100:
        x1 = 0.02
        x2 = 0.4
        z1 = -1.1682941853606887
    elif num_moments==40:
        x1 = 0.042
        x2 = 0.6
        z1 = -1.1145407146104997
    elif num_moments==20:
        x1 = 0.07
        x2 = 0.8
        z1 = -1.0577828186946745
    z2 = pi/2    

    x_op  = standard_operations.power_pulse_on_specific_directions(power=1, indices=[0])
    x2_op = standard_operations.power_pulse_on_specific_directions(power=2, indices=[0])
    z_op  = standard_operations.power_pulse_on_specific_directions(power=1, indices=[2])
    z2_op = standard_operations.power_pulse_on_specific_directions(power=2, indices=[2])
    
    operations = [x2_op, z_op, x_op, z2_op, x_op, z2_op ]
    theta      = [x1,    z1,   x2,   z2,    x2,   z2    ]
        
    # Apply:
    final_state = coherent_control.custom_sequence(state=initial_state, theta=theta, operations=operations, movie_config=movie_config)
    
    # Plot
    visuals.plot_plain_wigner(final_state)
    
    plt = visuals.plot_wigner_bloch_sphere(final_state, num_points=150, view_elev=-50)
    fig = plt.axes.figure
    fig.suptitle("GKP State", fontsize=16)

    visuals.draw_now()
    print("Movie is ready in folder 'video' ")

    # plot
    '''
    '''    

def _test_custom_sequence():
    # Const:
    num_moments:int=40
    active_movie_recorder:bool=False
    fps=10
    
    
    num_transition_frames=20 if active_movie_recorder else 0
    
    
    # define score function:
    from common_cost_functions import fidelity_to_cat
    fidelity_to_cat_ = fidelity_to_cat(num_atoms=40, num_legs=2, phase=np.pi/2)


    ## Define operations:
    coherent_control = CoherentControl(num_atoms=num_moments)
    standard_operations : CoherentControl.StandardOperations = coherent_control.standard_operations(num_intermediate_states=num_transition_frames)

    rotation  = standard_operations.power_pulse_on_specific_directions(power=1, indices=[0, 1, 2])
    squeezing = standard_operations.power_pulse_on_specific_directions(power=2, indices=[2])
    
    pi = np.pi
    pi_half = pi/2
    
    theta = [
        +pi_half , +0.00 , -0.00 , 
        +1.0  ,
        -pi_half , -0.00 , +0.00 
    ]
    

    operations = [
        rotation, squeezing, rotation
    ]
    
    initial_state = Fock.ground_state_density_matrix(num_moments)
    
    # Apply:
    final_state = coherent_control.custom_sequence(state=initial_state, theta=theta, operations=operations)
    # visuals.plot_wigner_bloch_sphere(final_state, num_points=200, view_elev=-55, alpha_min=1)
    visuals.plot_plain_wigner(final_state)
    visuals.draw_now()
    # visuals.save_figure()
    print("Done.")
    
if __name__ == "__main__":    
    np_utils.fix_print_length()

    # _test_pulse_in_steps()
    # _test_record_sequence()
    # _test_power_pulse()
    # _test_goal_gkp()
    _test_custom_sequence()
    # _show_analitical_gkp()

    print("Done.")

    
