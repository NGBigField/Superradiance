# ==================================================================================== #
# |                                   Imports                                        | #
# ==================================================================================== #

# Everyone needs numpy:
import numpy as np
from numpy import pi

# import our helper modules
from utils import (
    assertions,
    numpy_tools as np_utils,
    visuals,
    saveload,
    strings,
)

# For defining coherent states:
from fock import Fock

# For coherent control
from coherentcontrol import CoherentControl, _DensityMatrixType

# for plotting
import matplotlib.pyplot as plt  # for plotting test results:

# For optimizations:
from optimization import learn_custom_operation, LearnedResults, minimize, learn_single_op

# ==================================================================================== #
#|                                helper functions                                    |#
# ==================================================================================== #

def _get_final_state(
    ground_state:_DensityMatrixType, coherent_control:CoherentControl,
    x1, x2
):
    # Basic pulses:
    Sy = coherent_control.s_pulses.Sy
    Sy2 = Sy@Sy
    Sz = coherent_control.s_pulses.Sz

    # Act with pulses:
    rho = ground_state
    rho = coherent_control.pulse_on_state(rho, x=x1, power=2) 
    rho, z1 = learn_single_op(rho, Sz, Sy2)

    z2 = pi/2

    rho = coherent_control.pulse_on_state(rho, x=x2, power=1)
    rho = coherent_control.pulse_on_state(rho, z=z2, power=2)

    rho = coherent_control.pulse_on_state(rho, x=x2, power=1)
    rho = coherent_control.pulse_on_state(rho, z=z2, power=2)

    visuals.plot_light_wigner(rho)
    visuals.draw_now()

    return rho

# ==================================================================================== #
#|                                   main tests                                       |#
# ==================================================================================== #
    
def _example_gkp2(
    num_moments = 40
):

    # Define the basics:
    ground_state = Fock.ground_state_density_matrix(num_moments=num_moments)    
    coherent_control = CoherentControl(num_moments=num_moments)
    final_state = lambda x1, x2 : _get_final_state(ground_state, coherent_control, x1, x2)

    # Derive size-specific variables:
    if num_moments==20:
        x1 = 0.02
        x2 = 0.8
    elif num_moments==40:
        x1 = 0.09
        x2 = 0.2
    elif num_moments==100:
        x1 = 0.02
        x2 = 0.4
    else:
        raise ValueError(f"This number is not supported. num_moments={num_moments}")

    # Act with Pulses:
    x1, x2 = 0.072, 0.3
    final_state(x1, x2)
    print("Finished.")
    


def _example_gkp(
    num_moments = 100
):
    ground_state = Fock.ground_state_density_matrix(num_moments=num_moments)    
    coherent_control = CoherentControl(num_moments=num_moments)
    Sx = coherent_control.s_pulses.Sx
    Sx2 = Sx@Sx
    Sy = coherent_control.s_pulses.Sy
    Sy2 = Sy@Sy

    rho = ground_state
    rho = coherent_control.pulse_on_state(rho, x=0.03, power=2) 

    def cost_func(theta:np.ndarray)->float:
        z = theta[0]
        final_state = coherent_control.pulse_on_state(rho, z=z, power=1) 
        x2_proj = np.trace(final_state@Sy2)
        return x2_proj

    result = minimize(cost_func, [0])
    z = result.x[0]
    rho = coherent_control.pulse_on_state(rho, z=z, power=1)

    x = 0.4
    z = pi/2

    rho = coherent_control.pulse_on_state(rho, x=x, power=1)
    rho = coherent_control.pulse_on_state(rho, z=z, power=2)

    # rho = coherent_control.pulse_on_state(rho, x=z, power=1)

    rho = coherent_control.pulse_on_state(rho, x=x, power=1)
    rho = coherent_control.pulse_on_state(rho, z=z, power=2)

    visuals.plot_light_wigner(rho)
    visuals.draw_now()


    print("Finished.")

if __name__ == "__main__":
    _example_gkp2()
    print("Finished main.")

