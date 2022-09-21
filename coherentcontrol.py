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
    List,
)

# import our helper modules
from utils import (
    assertions,
    numpy as np_utils
)

# For measuring time:
import time

# For visualizations:
import matplotlib.pyplot as plt  # for plotting test results:
from light_wigner.main import visualize_light_from_atomic_density_matrix
from light_wigner.distribution_functions import Atomic_state_on_bloch_sphere
from visuals import plot_city

# For OOP:
from dataclasses import dataclass

# ==================================================================================== #
# |                                  Constants                                       | #
# ==================================================================================== #
OPT_METHOD = 'COBYLA'




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
        Sx, Sy, Sz = S_mats(N)
        self.Sx = Sx 
        self.Sy = Sy 
        self.Sz = Sz
    

class CoherentControl():

    def __init__(self, N:int=2) -> None:
        # Keep basic properties:        
        self._N = N
        # define basic pulses:
        self.s_pulses = SPulses(N)

    def pulse(self, x:float=0.0, y:float=0.0, z:float=0.0) -> np.matrix:
        Sx = self.s_pulses.Sx 
        Sy = self.s_pulses.Sy 
        Sz = self.s_pulses.Sz
        return pulse(x,y,z, Sx,Sy,Sz)

    def pulse_on_state(self, state:np.matrix, x:float=0.0, y:float=0.0, z:float=0.0) -> np.matrix :
        p = self.pulse(x,y,z)
        final_state = p * state * p.getH()
        return final_state

    @property
    def N(self) -> int:
        return self._N
    @N.setter
    def N(self, val:Any) -> None:
        self._N = val
        self.s_pulses = SPulses(val)


# ==================================================================================== #
# |                                   main                                           | #
# ==================================================================================== #


#TODO: 
#   M is from -J to J   
#   but we use the index 
#   m from 0 to N
""" _summary_
S_+ = ( 0   1 )
      ( 0   0 )

S_- = ( 0   0 )
      ( 1   0 )

S_X = ( 0   1 )
      ( 1   0 )

S_Z = ( 1   0 )
      ( 0  -1 )
"""

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
    from schrodinger_evolution import init_state, Params, CommonStates    
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

    """ 

    cost functions:

    * Even\odd cat states (atomic density matrix)  (poisonic dist. pure state as a |ket><bra| )

    * purity measure:  trace(rho^2)
        1 - if pure
        1/N - maximally not pure 

    * BSV light
    """

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

    rho_final = _apply_pulse_on_initial_state(c)
    np_utils.print_mat(rho_final)

    # visualizing light:
    title = f" rho "
    plot_city(rho_final, title=title)
    plt.show()
    # rho_final = np.array( rho_final.tolist() )
    # visualize_light_from_atomic_density_matrix(rho_final, N)

    # # visualizing matter:
    # Atomic_state_on_bloch_sphere.Wigner_BlochSphere()  # use this. this is better.


    


if __name__ == "__main__":    

    np_utils.fix_print_length()
    # _test_M_of_m()
    # _test_s_mats()
    _test_pi_pulse()
    print("Done.")


    visualize_light_from_atomic_density_matrix(1,2)
    