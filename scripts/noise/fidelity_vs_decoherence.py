# ==================================================================================== #
#| Imports:                                         
# ==================================================================================== #
if __name__ == "__main__":
    import sys, pathlib
    sys.path.append(
        pathlib.Path(__file__).parent.parent.parent.__str__()
    )
    
# For type annotations:
from typing import Callable
# Import useful types:
from algo.optimization import BaseParamType
from algo.coherentcontrol import Operation

# Basic algo mechanisms:
from algo.coherentcontrol import CoherentControl
from physics.famous_density_matrices import ground_state, gkp_state, cat_state

# for numerics:
import numpy as np
import scipy as sc
from qutip import jmat,Options

# Our best optimized results:    
from scripts.optimize.cat4_i     import best_sequence_params as cat4_params
from scripts.optimize.cat2_i     import best_sequence_params as cat2_params
from scripts.optimize.gkp_hex    import best_sequence_params as gkp_hex_params
from scripts.optimize.gkp_square import best_sequence_params as gkp_square_params

# Cost function:
from algo.common_cost_functions import fidelity_to_cat, fidelity_to_gkp

# for plotting:
from utils.visuals import plot_matter_state, plot_wigner_bloch_sphere, plot_plain_wigner, ViewingAngles, BlochSphereConfig, save_figure, draw_now
from utils import assertions, saveload
import matplotlib.pyplot as plt

# for printing progress:
from utils import strings

# for enums:
from enum import Enum, auto

# for sleeping:
from time import sleep

# For emitted light calculation 
from physics.emitted_light_approx import main as calc_emitted_light

# For writing results to file:
from csv import DictWriter

import itertools
import functools

# Qutip stuff
from qutip.piqs import Dicke, dicke, jspin, dicke_blocks
from qutip import steadystate, Qobj, mesolve

from algo.metrics import fidelity


class StateType(Enum):
    GKPHex = auto()
    GKPSquare = auto()
    Cat2 = auto()
    Cat4 = auto()


def _get_movie_config(
    create_movie:bool, num_transition_frames:int, temp_dir_name:str
) -> CoherentControl.MovieConfig:
    # Basic data:
    fps=30
    
    bloch_sphere_config = BlochSphereConfig(
        alpha_min=0.2,
        resolution=250,
        viewing_angles=ViewingAngles(
            elev=-45
        )
    )
    
    # Movie config:
    movie_config=CoherentControl.MovieConfig(
        active=create_movie,
        show_now=False,
        num_freeze_frames=fps//2,
        fps=fps,
        bloch_sphere_config=bloch_sphere_config,
        num_transition_frames=num_transition_frames,
        temp_dir_name=temp_dir_name
    )
    
    return movie_config


DONT_CREATE_MOVIE_CONFIG = _get_movie_config(False, 0, "")                


def _get_best_params(
    type_:StateType, 
    num_atoms:int,
    num_intermediate_states:int
) -> tuple[
    list[BaseParamType],
    list[Operation]
]:
    if type_ is StateType.GKPHex:
        return gkp_hex_params(num_atoms, num_intermediate_states=num_intermediate_states)
    elif type_ is StateType.GKPSquare:
        return gkp_square_params(num_atoms, num_intermediate_states=num_intermediate_states)
    elif type_ is StateType.Cat4:
        return cat4_params(num_atoms, num_intermediate_states=num_intermediate_states)
    elif type_ is StateType.Cat2:
        return cat2_params(num_atoms, num_intermediate_states=num_intermediate_states)
    else:
        raise ValueError(f"Not an option '{type_}'")


def _get_cost_function(type_:StateType, num_atoms:int) -> Callable[[np.matrix], float]:
    if type_ is StateType.GKPHex:
        return fidelity_to_gkp(num_atoms=num_atoms, gkp_form="hex")
    elif type_ is StateType.GKPSquare:
        return fidelity_to_gkp(num_atoms=num_atoms, gkp_form="square")        
    elif type_ is StateType.Cat4:
        return fidelity_to_cat(num_atoms=num_atoms, num_legs=4, phase=np.pi/4)
    elif type_ is StateType.Cat2:
        return fidelity_to_cat(num_atoms=num_atoms, num_legs=2, phase=np.pi/2)        
    else:
        raise ValueError(f"Not an option '{type_}'")


def _get_type_inputs(
    state_type:StateType, num_atoms:int, num_intermediate_states:int
) -> tuple[
    CoherentControl,
    np.matrix,
    list[float],
    list[Operation],
    Callable[[np.matrix], float]
]:
    # Get all needed data:
    params, operations = _get_best_params(state_type, num_atoms, num_intermediate_states)
    initial_state = ground_state(num_atoms=num_atoms)
    coherent_control = CoherentControl(num_atoms=num_atoms)
    cost_function = _get_cost_function(type_=state_type, num_atoms=num_atoms)
    
    # derive theta:
    theta = [param.get_value() for param in params]
    
    return coherent_control, initial_state, theta, operations, cost_function

@functools.cache
def _op_hamiltonian_from_op_index(N:int, x:float, y:float, z:float,power:int):
    [jx, jy, jz] = jspin(N)
    if power == 1:
        return x*jx + y *jy + z*jz
    elif power == 2:
        return x*jx*jx + y *jy*jy

def state_generation_with_dephasing(gamma_d:float,
    N:int = 40,
    state_type:StateType = StateType.Cat2
):

    ## get
    coherent_control, initial_state, thetas, operations, cost_function = _get_type_inputs(state_type=state_type, num_atoms=N, num_intermediate_states=0)
    # num_steps = sum([1 for op in operations if op.name=="squeezing"])    
    # matter_state = coherent_control.custom_sequence(state=initial_state, theta=theta, operations=operations, movie_config=DONT_CREATE_MOVIE_CONFIG)

    ## Global:
    ground = dicke(N, N/2, -N/2)
    rho = ground  # First time, we start from the ground
    Omega=1 #GHz
    Chi=0.03# GHz
    i=0
    ## Iterations for each pulse
    while i < len(thetas):
        # part 1 rotation
        r = (thetas[i]**2 +  thetas[i+1]**2 +  thetas[i+2]**2)**0.5
        op_hamiltonian = _op_hamiltonian_from_op_index(N,Omega*thetas[i]/r,Omega*thetas[i+1]/r,Omega*thetas[i+2]/r,power=1)
        # Time is the absolute value of theta, and the hamiltonian is depnadnant on the sign of it:
        system = Dicke(N, hamiltonian = op_hamiltonian, emission = 0, dephasing = gamma_d)  #TODO: Add noise
        tlist = np.linspace(0, r/Omega, 10)

        # solve:
        liouv = system.liouvillian()
        res = mesolve(liouv, rho, tlist=tlist)

        # Get the new state
        rho = res.states[-1]

        i = i+3

        # part 2 squeezing
        if i == len(thetas):
            break
        r = (thetas[i]**2 +  thetas[i+1]**2)**0.5
        op_hamiltonian = _op_hamiltonian_from_op_index(N,Chi*thetas[i]/r,Chi*thetas[i+1]/r,0,power=2)
        # Time is the absolute value of theta, and the hamiltonian is depnadnant on the sign of it:
        system = Dicke(N, hamiltonian = op_hamiltonian, emission = 0, dephasing = gamma_d)  #TODO: Add noise
        tlist = np.linspace(0, r/Chi, 10)
        i = i+2

        # solve:
        liouv = system.liouvillian()
        res = mesolve(liouv, rho, tlist=tlist)
        #,options=Options(nsteps=10000)
        # Get the new state
        rho = res.states[-1]

 # final rotation
    op_hamiltonian = _op_hamiltonian_from_op_index(N,0,np.pi,0,power=1)
    system = Dicke(N, hamiltonian = op_hamiltonian, emission = 0, dephasing = 0)  #TODO: Add noise
    tlist = np.linspace(0, 1, 10)
    liouv = system.liouvillian()
    res = mesolve(liouv, rho, tlist=tlist)

    # Get the new state
    rho = res.states[-1]

    blocks = dicke_blocks(rho)
    block0 = blocks[0]

    return block0

def optimal_fidelity(N,rho1,rho2):
    N_theta,N_phi = 100,200
    thetas=np.linspace(0,np.pi,N_theta)
    phis=np.linspace(0,2*np.pi,N_phi)
    [jx,jy,jz]=jmat(N/2)
    dphi = 2*np.pi/N_phi
    dtheta = np.pi/N_theta
    max_fid = 0
    for theta in thetas:
        for phi in phis:
            rho1 = sc.linalg.expm(-1j*jz*dphi)@rho1@sc.linalg.expm(1j*jz*dphi)
            fid = fidelity(rho1,rho2)
            max_fid = max(max_fid,fid)
        rho1 = sc.linalg.expm(-1j*jx*dtheta)@rho1@sc.linalg.expm(1j*jx*dtheta)
    return max_fid

def main():
    N=40
    N_GAMMA = 30
    GAMMA_D_MIN = -6
    GAMMA_D_MAX = -3
    # 2 - cat 
    gammas_dephasing = np.logspace(GAMMA_D_MIN,GAMMA_D_MAX,N_GAMMA)
    ideal_state_cat = cat_state(N,2,3,np.pi/2)
    #ideal_state_gkp=gkp_state(N,"square")
    
    fidelities = np.zeros(N_GAMMA)

    for i, gamma_d in enumerate(gammas_dephasing):
        print(str(i+1)+ '/' + str(len(gammas_dephasing)))
        rho_out = state_generation_with_dephasing(gamma_d,state_type=StateType.Cat2)


        fidelities[i] = optimal_fidelity(N,rho_out, ideal_state_cat)
        print('fidelity cat=' + str(fidelities[i]))
        np.save('fidelity_2cat',fidelities)
        np.save('gammad_2cat',gammas_dephasing)


def main2():
    N=40
    N_GAMMA = 30
    GAMMA_D_MIN = -6
    GAMMA_D_MAX = -3
    gammas_dephasing = np.logspace(GAMMA_D_MIN,GAMMA_D_MAX,N_GAMMA)
    ideal_state_gkp=gkp_state(N,"square")
    
    fidelities = np.zeros(N_GAMMA)

    for i, gamma_d in enumerate(gammas_dephasing):
        print(str(i+1)+ '/' + str(len(gammas_dephasing)))
        rho_out = state_generation_with_dephasing(gamma_d,state_type=StateType.GKPSquare)


        fidelities[i] = optimal_fidelity(N,rho_out, ideal_state_gkp)
        print('fidelity gkp=' + str(fidelities[i]))
        np.save('fidelity_gkp',fidelities)
        np.save('gammad_gkp',gammas_dephasing)

if __name__ == "__main__":
    main()
    main2()





