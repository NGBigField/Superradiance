# ==================================================================================== #
# |                                 Imports                                          | #
# ==================================================================================== #

# Everyone needs numpy:
import numpy as np

# For matrix exponential: 
from scipy.linalg import expm

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
    TypeVar,
)

# import our helper modules
from utils import (
    args,
    assertions,
    numpy_tools as np_utils,
    visuals,
    strings,
    decorators,
)

# For measuring time:
import time

# For visualizations:
import matplotlib.pyplot as plt  # for plotting test results:
from matplotlib.axes import Axes  # for type hinting:

# For OOP:
from dataclasses import dataclass, field

# For simulating state decay:
from light_wigner.main import decay
from evolution import (
    evolve,
    Params as evolution_params,
)

# for copying input:
from copy import deepcopy

# For checking on real fock states:
from quantum_states.fock import Fock

# ==================================================================================== #
# |                                  Constants                                       | #
# ==================================================================================== #
OPT_METHOD : Final = 'COBYLA'

# ==================================================================================== #
# |                                 Helper Types                                     | #
# ==================================================================================== #
# ==================================================================================== #
# |                                 Helper Types                                     | #
# ==================================================================================== #
@dataclass
class _PulseSequenceParams():
    xyz : Tuple[float]
    pause : float

_DensityMatrixType = np.matrix


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
        self.Sx = Sx 
        self.Sy = Sy 
        self.Sz = Sz

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
    
    @dataclass
    class Config():
        active : bool = False
        show_now : bool = True
        fps : int = 5
        num_transition_frames : int = 10
        num_freeze_frames : int = 5
        bloch_sphere_resolution : int = 25
    
    def __init__(
        self, 
        initial_state : _DensityMatrixType,
        score_string_function : Optional[Callable[[np.matrix], str]] = None,
        config : Optional[Config] = None,
    ) -> None:
        # Base properties:        
        self.config : SequenceMovieRecorder.Config = args.default_value(config, default_factory=SequenceMovieRecorder.Config )
        self.video_recorder : visuals.VideoRecorder = visuals.VideoRecorder(fps=self.config.fps)
        if self.config.active:
            self.figure_object : visuals.MatterStatePlot = visuals.MatterStatePlot(
                block_sphere_resolution=self.config.bloch_sphere_resolution,
                initial_state=initial_state
            )            
        else:
            self.figure_object = None
        self.score_string_function : Optional[Callable[[np.matrix], str]] = score_string_function
        
    def _record_single_state(
        self,
        state : _DensityMatrixType,
        duration : int,  # number of repetitions of the same frame:
        title : Optional[str]=None,
        similarity_str : Optional[str] = None,
    ) -> None:
        self.figure_object.update(state, title=title, show_now=self.config.show_now)
        self.video_recorder.capture(fig=self.figure_object.figure, duration=duration)
        
    def record_transition(
        self, 
        transition_states:np.matrix, 
        title:str, 
    ) -> None:
        # Check inputs:
        if not self.is_active:
            return  # We don't want to record a video
        final_state = transition_states[-1]
        # Add similarity string:
        if self.score_string_function is None:
            similarity_str = None
        else:
            similarity_str = self.score_string_function(final_state)
        # Capture shots: (transition and freezed state)
        for transition_state in transition_states:
            self._record_single_state(transition_state, title=title, duration=1 )
        self._record_single_state(final_state, title=None, duration=self.config.num_freeze_frames, similarity_str=similarity_str)
        # Keep info for next call:
        self.last_state = deepcopy(transition_states)
        
    def write_video(self) -> None:
        if not self.is_active:
            return
        self.video_recorder.write_video()

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
        num_moments:int,
        gamma:float=1.0,
    ) -> None:
        # Keep basic properties:        
        self._num_moments = num_moments
        self.gamma = gamma
        # define basic pulses:
        self.s_pulses = SPulses(num_moments)

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

    def _pulse(self, x:float=0.0, y:float=0.0, z:float=0.0) -> np.matrix:
        Sx = self.s_pulses.Sx 
        Sy = self.s_pulses.Sy 
        Sz = self.s_pulses.Sz
        return pulse(x,y,z, Sx,Sy,Sz)

    def _state_decay_old_iterative_method(
        self,
        state:np.matrix, 
        time:float, 
        time_steps:Optional[int]=None,
    )->np.matrix:
        # Complete missing inputs:
        time_steps = args.default_value(time_steps, self._default_state_decay_resolution)    
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
    def pulse_on_state_final_state(self, state:_DensityMatrixType, x:float=0.0, y:float=0.0, z:float=0.0) -> _DensityMatrixType: 
        return self.pulse_on_state(state=state, num_intermediate_states=0, x=x, y=y, z=z)[-1]

    def pulse_on_state(self, state:_DensityMatrixType, num_intermediate_states:int=0, x:float=0.0, y:float=0.0, z:float=0.0) -> List[_DensityMatrixType]: 
        # Check input:
        num_intermediate_states_error_msg = f"`num_intermediate_states` must be a non-negative number."
        num_intermediate_states = assertions.integer(num_intermediate_states)
        if num_intermediate_states > 0:
            num_divides = num_intermediate_states
        elif num_intermediate_states == 0:
            num_divides = 1
        else:
            raise ValueError(num_intermediate_states_error_msg)
        # Divide requested pulse into fragments
        frac_x = x / num_divides
        frac_y = y / num_divides
        frac_z = z / num_divides
        p = self._pulse(frac_x, frac_y, frac_z)
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

    def state_decay_final_state(self, state:_DensityMatrixType, time:float, time_steps_resolution:Optional[int]=None,) -> _DensityMatrixType: 
        return self.state_decay(state=state, time=time, num_intermediate_states=0, time_steps_resolution=time_steps_resolution)[-1]        

    def state_decay(
        self, 
        state:_DensityMatrixType, 
        time:float, 
        num_intermediate_states:int=0,
        time_steps_resolution:Optional[int]=None
    ) -> List[_DensityMatrixType] :
        # Check inputs:
        assertions.density_matrix(state, robust_check=True)  # allow matrices to be non-PSD or non-Hermitian
        num_intermediate_states = assertions.integer(num_intermediate_states, reason=f"`num_intermediate_states` must be a non-negative integer")
        assert time>0, f"decay time must be a positive number. got {time}"
        # Complete missing inputs:
        time_steps_resolution = args.default_value(time_steps_resolution, 10001)        
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
        num2str = lambda x : strings.formatted(x, width=5, decimals=5)
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
                transition_states = self.pulse_on_state(state=crnt_state, x=x, y=y, z=z, num_intermediate_states=num_intermediate_states)
                sequence_recorder.record_transition(transition_states, f"Pulse = [{num2str(x)}, {num2str(y)}, {num2str(z)}]")
                crnt_state = transition_states[-1]
            if t != 0:
                transition_states = self.state_decay(state=crnt_state, time=t, num_intermediate_states=num_intermediate_states)
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
    coherent_control = CoherentControl(num_moments=num_moments)
    final_state = coherent_control.coherent_sequence(state=initial_state, theta=theta, record_movie=True, movie_config=movie_config )
    print("Movie is ready in folder 'video' ")


def _test_pulse_in_steps():
    # Define params:
    num_moments:int=4   
    num_steps:int=20
    fps:int=5
    # Init state:
    initial_state = Fock(0).to_density_matrix(num_moments=num_moments)
    coherent_control = CoherentControl(num_moments=num_moments)
    # Apply pulse:
    all_pulse_states = coherent_control.pulse_on_state(state=initial_state, num_intermediate_states=num_steps, x=np.pi )
    # Apply decay time:
    all_decay_states = coherent_control.state_decay(state=all_pulse_states[-1], num_intermediate_states=num_steps, time=0.5)
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

if __name__ == "__main__":    
    np_utils.fix_print_length()

    # _test_pulse_in_steps()
    _test_record_sequence()

    print("Done.")

    


#NOTE:

""" 

possible cost functions:

* Even\odd cat states (atomic density matrix)  (poisonic dist. pure state as a |ket><bra| )

* purity measure:  trace(rho^2)
    1 - if pure
    1/N - maximally not pure 

* BSV light
"""