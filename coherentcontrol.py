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
    Tuple,
    Optional,
    List,
    ClassVar,
    Final,
)

# import our helper modules
from utils import (
    assertions,
    numpy_tools as np_utils,
    visuals,
)

# For measuring time:
import time

# For visualizations:
import matplotlib.pyplot as plt  # for plotting test results:
from light_wigner.main import visualize_light_from_atomic_density_matrix
from light_wigner.distribution_functions import Atomic_state_on_bloch_sphere

# For OOP:
from dataclasses import dataclass

# For simulating state decay:
from evolution import (
    evolve,
    Params as evolution_params,
)

# for copying input:
from copy import deepcopy

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
        
def _deal_params(theta:List[float]) -> List[_PulseSequenceParams] :
    # Constants:
    NUM_PULSE_PARAMS : Final = 4
    # Check length:
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
    Sx = D_plus + D_minus 
    Sy = -1j*D_plus + 1j*D_minus 
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
    

class CoherentControl():

    # Class Attributes:
    _default_state_decay_resolution : ClassVar[int] = 1000

    # ==================================================== #
    #|                    constructor                     |#
    # ==================================================== #

    def __init__(self, max_state_num:int) -> None:
        # Keep basic properties:        
        self._max_state_num = max_state_num
        # define basic pulses:
        self.s_pulses = SPulses(max_state_num)
        # add default class properties:
        self.state_decay_resolution : int = CoherentControl._default_state_decay_resolution
    
    # ==================================================== #
    #|                   inner functions                  |#
    # ==================================================== #

    def _pulse(self, x:float=0.0, y:float=0.0, z:float=0.0) -> np.matrix:
        Sx = self.s_pulses.Sx 
        Sy = self.s_pulses.Sy 
        Sz = self.s_pulses.Sz
        return pulse(x,y,z, Sx,Sy,Sz)

    # ==================================================== #
    #|                 declared functions                 |#
    # ==================================================== #

    def pulse_on_state(self, state:np.matrix, x:float=0.0, y:float=0.0, z:float=0.0) -> np.matrix :
        p = self._pulse(x,y,z)
        final_state = p * state * p.getH()
        return final_state
    
    def state_decay(self, state:np.matrix, time:float, time_steps:Optional[int]=None) -> np.matrix :
        # Complete missing inputs:
        if time_steps is None:
            time_steps = self.state_decay_resolution
        # Check inputs:
        assertions.integer(time_steps)
        assertions.density_matrix(state)
        # Params:
        params = evolution_params(
            N = self.max_state_num,
            dt=time/time_steps
        )
        # Init state:
        crnt_state = deepcopy(state) 
        for step in range(time_steps):
            crnt_state = evolve( rho=crnt_state, params=params )
        return crnt_state

    def coherent_sequence(self, state:np.matrix, theta:List[float]) -> np.matrix :
        """coherent_sequence Apply sequence of coherent pulses separated by state decay.

        The length of the `theta` is 4*p - 1 
        where `p` is the the number of pulses
        3*p x,y,z parameters for p pulses.
        p-1 decay-time parameters between each pulse.

        Args:
            state (np.matrix): initial density-matrix
            theta (List[float]): parameters.

        Returns:
            np.matrix: final density-matrix
        """
        # Check and prepare inputs:
        assertions.density_matrix(state)
        params = _deal_params(theta)
        crnt_state = deepcopy(state)

        # iterate:
        for pulse_params in params:
            # Unpack parans:
            x = pulse_params.xyz[0]
            y = pulse_params.xyz[1]
            z = pulse_params.xyz[2]
            pause = pulse_params.pause
            # Apply pulse and delay:
            crnt_state = self.pulse_on_state(state=crnt_state, x=x, y=y, z=z)
            if pause == 0: continue
            crnt_state = self.state_decay(state=crnt_state, time=pause)
        
        # End:
        return crnt_state
        

    # ==================================================== #
    #|                  static methods                    |#
    # ==================================================== #        
    @staticmethod
    def num_params_for_pulse_sequence(num_pulses:int) -> int:
        assertions.integer(num_pulses)
        return num_pulses*4-1

    # ==================================================== #
    #|                  setters\getters                   |#
    # ==================================================== #        

    @property
    def max_state_num(self) -> int:
        return self._max_state_num
    @max_state_num.setter
    def max_state_num(self, val:Any) -> None:
        self._max_state_num = val
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


def _test_decay(max_state_num:int=4, decay_time:float=1.00):
    # Specific imports for test:
    from quantum_states.fock import Fock

    # Constants:
    pi_pulse_x_value = 1.56
    
    # Prepare state:
    coherent_control = CoherentControl(max_state_num=max_state_num)
    zero_state = Fock.create_coherent_state(alpha=0, max_num=max_state_num).to_density_matrix(max_num=max_state_num)
    initial_state = coherent_control.pulse_on_state(state=zero_state, x=pi_pulse_x_value )

    # Let evolve:
    final_state = coherent_control.state_decay(state=initial_state, time=decay_time)

    # Plot:
    visuals.plot_city(final_state)


def _test_coherent_sequence(max_state_num:int=4, num_pulses:int=3):
    # Specific imports for test:
    from quantum_states.fock import Fock

    # init params:
    num_params = CoherentControl.num_params_for_pulse_sequence(num_pulses=num_pulses)
    theta = list(range(num_params))
    initial_state = Fock.create_coherent_state(alpha=0, max_num=max_state_num).to_density_matrix(max_num=max_state_num)
    
    # Apply sequence:
    coherent_control = CoherentControl(max_state_num=max_state_num)
    final_state = coherent_control.coherent_sequence(state=initial_state, theta=theta)

    # Plot:
    visuals.plot_city(final_state)


if __name__ == "__main__":    

    np_utils.fix_print_length()
    # _test_M_of_m()
    # _test_s_mats()
    # _test_pi_pulse(N=4, MAX_ITER=10)
    # _test_decay()
    _test_coherent_sequence()
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