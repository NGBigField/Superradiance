import qutip
import numpy as np
from numpy import pi
import matplotlib.pyplot as plt
from coherentcontrol import CoherentControl, Operation
from fock import Fock
import metrics
from typing import Callable


def initial_guess(num_moments:int):
    # Prepare coherent control
    coherent_control = CoherentControl(num_moments=num_moments)
    # Operations:
    operations = [
        Operation(
            num_params=0, 
            function=lambda rho: coherent_control.pulse_on_state(rho, x=pi/2), 
            string="pi/2 Sx"
        )
    ]
    # init params:
    initial_state = Fock.ground_state_density_matrix(num_moments)
    # Apply:
    final_state = coherent_control.custom_sequence(state=initial_state, theta=None, operations=operations)
    return final_state

def goal_gkp_state(num_moments:int):
    # Get un-rotated state:
    psi = _goal_gkp_state_ket(num_moments)
    rho = qutip.ket2dm(psi).full()
    
    # Rotate the state:
    coherent_control = CoherentControl(num_moments=num_moments)
    # gkp = coherent_control.pulse_on_state(rho, x=pi/2)
    gkp = rho
    return gkp
    
def _goal_gkp_state_ket(
    num_moments:int,
    d_b = 10.0
):
    # Constants:
    alpha =  np.sqrt(pi/2)
    
    r = np.log( np.sqrt( d_b ) )
    m = round( (np.exp(r)**2) /pi )  # 3 when dB=10
    
    # Basic operators:
    n = num_moments + 1 
    S = qutip.squeeze(n, r)
    D_alpha = qutip.displace(n, alpha) + qutip.displace(n, -alpha)
    
    # Perform sequence:
    psi = qutip.basis(n, 0)
    psi = S * psi
    for _ in range(m):
        psi = D_alpha * psi    

    # Normalize:
    psi = psi.unit()

    return psi

def _test_plot_goal_gkp():
    num_moments = 40

    rho = goal_gkp_state(num_moments)

    from utils.visuals import plot_matter_state, plot_light_wigner

    plot_matter_state(rho)
    plot_light_wigner(rho)
    
    print("Done.")

def get_gkp_cost_function(num_moments:int)->Callable[[np.matrix], float]:
    # Define target:
    # target_4legged_cat_state = cat_state(num_moments=num_moments, alpha=3, num_legs=4).to_density_matrix()
    taget_state = goal_gkp_state(num_moments)
    # visuals.plot_matter_state(target_4legged_cat_state, block_sphere_resolution=200)
    def cost_function(final_state:np.matrix) -> float : 
        return -1 * metrics.fidelity(final_state, taget_state)       

    return cost_function

if __name__ == '__main__':
    _test_plot_goal_gkp()
    print("Done.")

