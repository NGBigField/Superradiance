from typing import Callable
import numpy as np
from algo.coherentcontrol import _DensityMatrixType
from algo.metrics import fidelity
from physics import gkp 
from physics.famous_density_matrices import cat_state
from utils import indices


__all__ = [
    "fidelity_to_gkp",
    "fidelity_to_cat",
    "fidelity_to_ghz",
    "weighted_ghz"
]

def fidelity_to_gkp(num_atoms:int, gkp_form:str="square")->Callable[[_DensityMatrixType], float] :
    return gkp.get_gkp_cost_function(num_atoms=num_atoms, form=gkp_form)

def fidelity_to_cat(num_atoms:int, num_legs:int, phase:float=0.0)->Callable[[_DensityMatrixType], float]:
    target = cat_state(num_atoms, alpha=3, num_legs=num_legs, phase=phase)
    def cost_func(rho:np.matrix)->float:
        return -1*fidelity(rho, target)
    return cost_func
    

def fidelity_to_ghz(initial_state:_DensityMatrixType) -> Callable[[_DensityMatrixType], float] :
    # Define cost function
    matrix_size = initial_state.shape[0]
    target_state = np.zeros(shape=(matrix_size,matrix_size))
    for i in [0, matrix_size-1]:
        for j in [0, matrix_size-1]:
            target_state[i,j]=0.5

    def cost_function(final_state:_DensityMatrixType) -> float : 
        return (-1) * fidelity(final_state, target_state)

    return cost_function

def weighted_ghz(initial_state:_DensityMatrixType) -> Callable[[_DensityMatrixType], float] :
    # Define cost function
    matrix_size = initial_state.shape[0]

    def cost_function(final_state:_DensityMatrixType) -> float : 
        total_cost = 0.0
        for i, j in indices.all_possible_indices((matrix_size, matrix_size)):
            element = final_state[i,j]
            if (i in [0, matrix_size-1]) and  (j in [0, matrix_size-1]):  # Corners:
                cost = np.absolute(element-0.5)  # distance from 1/2
            else: 
                cost = np.absolute(element)  # distance from 0
            total_cost += cost
        return total_cost

    return cost_function