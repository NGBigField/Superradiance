from physics.fock import Fock 
from physics.fock import cat_state as cate_state_ket
from physics.gkp import gkp_state
import numpy as np

__all__ = [
    "gkp_state",
    "ground_state",
    "fully_excited_state",
    "cat_state"
]

def ground_state(num_atoms:int)->np.matrix:
    return Fock.ground_state_density_matrix(num_atoms=num_atoms)

def fully_excited_state(num_atoms:int)->np.matrix:
    return Fock.excited_state_density_matrix(num_atoms=num_atoms)

def cat_state(num_atoms:int, num_legs:int, alpha:float, phase:float=0.0)->np.matrix:
    ket = cate_state_ket(num_atoms=num_atoms, alpha=alpha, num_legs=num_legs, phase=phase)
    return ket.to_density_matrix(num_moments=num_atoms)

    