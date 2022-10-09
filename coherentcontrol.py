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
)

# import our helper modules
from utils import (
    args,
    assertions,
    numpy_tools as np_utils,
    visuals,
    strings,
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
@dataclass
class _PulseSequenceParams():
    xyz : Tuple[float]
    pause : float

_DensityMatrixType = np.matrix

# ==================================================================================== #
# |                               Inner Functions                                    | #
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
        num_transition_frames : int = 10
        num_freeze_frames : int = 5
        fps : int = 5
        bloch_sphere_resolution : int = 100    
    
    def __init__(
        self, 
        is_active : bool, 
        score_string_function : Optional[Callable[[np.matrix], str]] = None,
        config : Optional[Config] = None,
    ) -> None:
        # Base properties:        
        self.is_active : bool = is_active
        self.config : SequenceMovieRecorder.Config = args.default_value(config, SequenceMovieRecorder.Config() )
        self.video_recorder : visuals.VideoRecorder = visuals.VideoRecorder(fps=self.config.fps)
        if is_active:
            self.figure_object : visuals.MatterStatePlot = visuals.MatterStatePlot(block_sphere_resolution=self.config.bloch_sphere_resolution)            
        else:
            self.figure_object = None
        self.score_string_function : Optional[Callable[[np.matrix], str]] = score_string_function
        # Initialized Tracking params:
        self.last_state : Union[np.matrix, None] = None
        
    def _record_single_shot(
        self,
        state : _DensityMatrixType,
        title : str,
        duration : int,  # number of repetitions of the same frame:
        similarity_str : Optional[str] = None,
    ) -> None:
        self.figure_object.update(state, title=title)
        self.video_recorder.capture(fig=self.figure_object.figure, duration=duration)
        
    def _transition_states(self, final_state:_DensityMatrixType) -> Generator[_DensityMatrixType, None, None]:
        # Check requirements:
        if self.last_state is None:  # Happens on first call to function
            return
        num_transitions = self.config.num_transition_frames
        if num_transitions <= 0:
            return
        # Keep basic info:
        first_state = self.last_state
        # Assert props:
        assert final_state.shape == first_state.shape
        # Create transition:
        diff_matrix = final_state - first_state
        addition_step_matrix = diff_matrix * (1/num_transitions) 
        for step in range(1, num_transitions+1):
            transition_state = first_state + step*addition_step_matrix
            yield transition_state
        
    def record_state(
        self, 
        state:np.matrix, 
        title:str, 
    ) -> None:
        # Check inputs:
        if not self.is_active:
            return  # We don't want to record a video
        # Add similarity string:
        if self.score_string_function is None:
            similarity_str = None
        else:
            similarity_str = self.score_string_function(state)
        # Capture shots: (transition and freezed state)
        for transition_state in self._transition_states(final_state=state):
            self._record_single_shot(transition_state, title, duration=1 )
        self._record_single_shot(state, title, duration=self.config.num_freeze_frames, similarity_str=similarity_str)
        # Keep info for next call:
        self.last_state = deepcopy(state)
        
    def write_video(self) -> None:
        if not self.is_active:
            return
        self.video_recorder.write_video()

class CoherentControl():   

    # Class Attributes:
    _default_state_decay_resolution : ClassVar[int] = 1000

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

    def _state_decay_iterative(
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

    def _state_decay_solve_ode(
        self,
        state:np.matrix, 
        time:float, 
        time_steps:Optional[int]=10001,
    )->np.matrix:
        # Complete missing inputs:
        time_steps = args.default_value(time_steps, 10001)        
        assertions.integer(time_steps)            
        # Convert matrix to ndarray:
        if isinstance(state, np.matrix):
            state = np.array(state)
        # Compute:
        rho_at_all_times = decay(
            rho=state,  
            delta_t=time,
            gamma=self.gamma,
            num_time_steps=time_steps
        )
        

    # ==================================================== #
    #|                 declared functions                 |#
    # ==================================================== #

    def pulse_on_state_fragmented(self, state:_DensityMatrixType, num_intermediate_states:int, x:float=0.0, y:float=0.0, z:float=0.0) -> List[_DensityMatrixType]: 
        num_intermediate_states = assertions.integer(num_intermediate_states)
        frac_x = x / num_intermediate_states
        frac_y = y / num_intermediate_states
        frac_z = z / num_intermediate_states
        p = self._pulse(frac_x, frac_y, frac_z)
        pH = p.getH()
        states : List[_DensityMatrixType] = []
        crnt_state = deepcopy(state)
        states.append(crnt_state)   # initial state
        for step in range(1, num_intermediate_states+1):            
            crnt_state = p * crnt_state * pH
            states.append(crnt_state)
        return states

    def pulse_on_state(self, state:_DensityMatrixType, x:float=0.0, y:float=0.0, z:float=0.0) -> _DensityMatrixType: 
        p = self._pulse(x,y,z)
        final_state = p * state * p.getH()
        return final_state
        
    def state_decay_fragmented(
        self, 
        state:_DensityMatrixType, 
        num_intermediate_states:int,
        time:float, 
        time_steps_resolution:Optional[int]=None,
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


    def state_decay(
        self, 
        state:_DensityMatrixType, 
        time:float, 
        time_steps_resolution:Optional[int]=None,
    ) -> _DensityMatrixType :
        res = self.state_decay_fragmented(state=state, num_intermediate_states=0, time=time,time_steps_resolution=time_steps_resolution)
        print(res)
        return res[-1]


    def coherent_sequence(
        self, 
        state:np.matrix, 
        theta: Union[ List[float], np.ndarray ],
        record_video : bool = False
    ) -> np.matrix :
        """coherent_sequence Apply sequence of coherent pulses separated by state decay.

        The length of the `theta` is 4*p - 1 
        where `p` is the the number of pulses
        3*p x,y,z parameters for p pulses.
        p-1 decay-time parameters between each pulse.

        Args:
            state (np.matrix): initial density-matrix
            theta (List[float]): parameters.
            record_video bool: Should the sequence be recorded as a video.

        Returns:
            np.matrix: final density-matrix
        """
        # Check and prepare inputs:
        assertions.density_matrix(state, robust_check=True)
        params = _deal_params(theta)
        crnt_state = deepcopy(state)

        # For sequence recording:
        sequence_recorder = SequenceMovieRecorder(is_active=record_video, config=SequenceMovieRecorder.Config(bloch_sphere_resolution=10))
        sequence_recorder.record_state(crnt_state, f"Initial-State")
        num2str = lambda x : strings.formatted(x, width=5, decimals=5)

        # iterate:
        for pulse_params in params:
            # Unpack parans:
            x = pulse_params.xyz[0]
            y = pulse_params.xyz[1]
            z = pulse_params.xyz[2]
            t = pulse_params.pause
            # Apply pulse and delay:
            if x != 0 or y != 0 or z != 0:
                crnt_state = self.pulse_on_state(state=crnt_state, x=x, y=y, z=z)
                sequence_recorder.record_state(crnt_state, f"Pulse = [{num2str(x)}, {num2str(y)}, {num2str(z)}]")
            if t != 0:
                crnt_state = self.state_decay(state=crnt_state, time=t)
                sequence_recorder.record_state(crnt_state, f"Decay-Time = {num2str(t)}")
        
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


def _test_s_mats():
    _print = np_utils.print_mat
    for N in [2,4]:
        Sx, Sy, Sz = S_mats(N)
        print(f"N={N}")
        print("\nSx:")
        _print(Sx)
        print("\nSy:")
        _print(Sy)
        print("\nSz:")
        _print(Sz)
        print( "\n\n" )

def _test_M_of_m():
    N = 6

    # Init:
    M_vec = []
    m_vec = []

    # Compute:
    J = _J(N)
    for m in range(N+1):
        M = _M(m,J)
        M_vec.append(M)
        m_vec.append(m)
        
    # plot:
    plt.plot(m_vec, M_vec)
    plt.title(f"N={N}, J={J}")
    plt.grid(True)
    plt.xlabel("m")
    plt.ylabel("M")
    plt.show()

def _test_pi_pulse(MAX_ITER:int=4, N:int=2):
    # Specific imports:
    from evolution import init_state, Params, CommonStates    
    from scipy.optimize import minimize  # for optimization:    

    # Define pulse:
    Sx, Sy, Sz = S_mats(N)
    _pulse = lambda x, y, z, c : pulse( x,y,z, Sx,Sy,Sz, c )
    _x_pulse = lambda c : _pulse(1,0,0,c)    

    # init:
    params = Params(N=N)
    rho_initial = init_state(params, CommonStates.Ground)
    rho_target  = init_state(params, CommonStates.FullyExcited)

    # Helper functions:
    def _apply_pulse_on_initial_state(c:float) -> np.matrix: 
        p = _x_pulse(c)
        rho_final = p * rho_initial * p.getH()
        return rho_final

    def _derive_cost_function(c:float) -> float :  
        rho_final = _apply_pulse_on_initial_state(c)
        diff = np.linalg.norm(rho_final-rho_target)
        cost = diff**2
        return cost

    def _find_optimum():
        initial_point = 0.00
        options = dict(
            maxiter = MAX_ITER
        )            
        # Run optimization:
        start_time = time.time()
        minimum = minimize(_derive_cost_function, initial_point, method=OPT_METHOD, options=options)
        finish_time = time.time()

        # Unpack results:
        run_time = finish_time-start_time
        print(f"run_time={run_time} [sec]")
        return minimum

    # Minimize:
    opt = _find_optimum()

    # Unpack results:    
    c = opt.x
    assert len(c)==1
    assert np.isreal(c)[0]
    print(f"pulse x = {c}")

    rho_final = _apply_pulse_on_initial_state(c)
    np_utils.print_mat(rho_final)

    # visualizing light:
    title = f" rho "
    visuals.plot_city(rho_final, title=title)
    plt.show()


def _test_decay(num_moments:int=4, decay_time:Optional[float]=None, gamma:float=1.00, initial_pulse_xyz:Optional[List[float]]=None):
    # fill missing:
    initial_pulse_xyz = args.default_value(initial_pulse_xyz, [1.56, 0, 0])
    decay_time = args.default_value(decay_time, np.log(2)/gamma)
    
    # Prepare state:
    coherent_control = CoherentControl(num_moments=num_moments, gamma=gamma)
    zero_state = Fock.create_coherent_state(alpha=0, num_moments=num_moments).to_density_matrix(num_moments=num_moments)
    initial_state = coherent_control.pulse_on_state(zero_state, *initial_pulse_xyz)

    # Let evolve:
    final_state1 = coherent_control.state_decay(state=initial_state, time=decay_time, method='iterative')
    final_state2 = coherent_control.state_decay(state=initial_state, time=decay_time, method='solve_ode')
    visuals.plot_city(final_state1)
    visuals.plot_city(final_state2)
    plt.show()



def _test_coherent_sequence(max_state_num:int=4, num_pulses:int=3):
    # init params:
    num_params = CoherentControl.num_params_for_pulse_sequence(num_pulses=num_pulses)
    theta = list(range(num_params))
    initial_state = Fock.create_coherent_state(alpha=0, num_moments=max_state_num).to_density_matrix(num_moments=max_state_num)
    
    # Apply sequence:
    coherent_control = CoherentControl(num_moments=max_state_num)
    final_state = coherent_control.coherent_sequence(state=initial_state, theta=theta)

    # Plot:
    visuals.plot_city(final_state)
    plt.show()

def _zero_state(max_state_num:int=4) -> Fock :
    return Fock.create_coherent_state(alpha=0, num_moments=max_state_num).to_density_matrix(num_moments=max_state_num)

def _test_complex_state(max_state_num:int=2):
    # Init:
    rho_initial = _zero_state(max_state_num=max_state_num)
    coherent_control = CoherentControl(num_moments=max_state_num)
    # Apply:
    final_state = coherent_control.pulse_on_state(
        rho_initial, 
        x = -0.1, 
        y =  0.3,
        z = -0.2
    )

    # Plot:
    np_utils.print_mat(final_state)
    visuals.plot_city(final_state)
    plt.show()
    print("Plotted.")

def _test_record_sequence():
    num_moments:int=2
    num_pulses:int=3
    # init params:
    num_params = CoherentControl.num_params_for_pulse_sequence(num_pulses=num_pulses)
    theta = np.random.random((num_params)).tolist()
    initial_state = Fock.create_coherent_state(num_moments=num_moments, alpha=0, output='density_matrix')
    # Apply sequence:
    visuals.plot_wigner_bloch_sphere(initial_state, num_points=100)
    # plot bloch sphere:
    coherent_control = CoherentControl(num_moments=num_moments)
    final_state = coherent_control.coherent_sequence(state=initial_state, theta=theta, record_video=True)
    print("Movie is ready in folder 'video' ")
    
def _test_record_pi_pulse():
    # Define params:
    num_moments:int=4
    decay_time = 2.0
    theta = [np.pi, 0, 0, decay_time, 0, 0, 0]
    # init state:
    initial_state = Fock(0).to_density_matrix(num_moments=num_moments)
    # plot bloch sphere:
    coherent_control = CoherentControl(num_moments=num_moments)
    final_state = coherent_control.coherent_sequence(state=initial_state, theta=theta, record_video=True)
    print("Movie is ready in folder 'video' ")
    
def _test_special_state():
    # Define params:
    num_moments:int=4   
    theta = [np.pi/2, 0, 0]
    # init state:
    initial_state = Fock(num_moments//2).to_density_matrix(num_moments=num_moments)
    # plot bloch sphere:
    coherent_control = CoherentControl(num_moments=num_moments)
    final_state = coherent_control.coherent_sequence(state=initial_state, theta=theta, record_video=True)

def _test_pulse_in_steps():
    # Define params:
    num_moments:int=4   
    num_steps:int=50
    fps:int=25
    # Init state:
    initial_state = Fock(num_moments//2).to_density_matrix(num_moments=num_moments)
    coherent_control = CoherentControl(num_moments=num_moments)
    # Apply pulse:
    all_pulse_states = coherent_control.pulse_on_state_fragmented(state=initial_state, num_intermediate_states=num_steps, x=np.pi/2 )
    # Apply decay time:
    all_decay_states = coherent_control.state_decay_fragmented(state=all_pulse_states[-1], num_intermediate_states=num_steps, time=0.5)
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
    # _test_M_of_m()
    # _test_s_mats()
    # _test_pi_pulse(N=4, MAX_ITER=10)
    # _test_decay(initial_pulse_xyz=[0.1, 0.8, -0.3])
    # _test_coherent_sequence()
    # _test_complex_state()
    # _test_record_sequence()
    # _test_record_pi_pulse()
    # _test_special_state()
    _test_pulse_in_steps()
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