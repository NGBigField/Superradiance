# ==================================================================================== #
# |                                   Imports                                        | #
# ==================================================================================== #

# Everyone needs numpy:
import numpy as np

# For typing hints:
from typing import (
    Any,
    Tuple,
    List,
    Union,
    Dict,
    Final,
)

# import our helper modules
from utils import (
    assertions,
    numpy_tools as np_utils,
    visuals,
    saveload,
    strings,
)

# For defining coherent states:
from quantum_states.fock import Fock

# For coherent control
from coherentcontrol import CoherentControl

# for plotting
import matplotlib.pyplot as plt  # for plotting test results:

# For optimizations:
from optimization import learn_specific_state, LearnedResults

# ==================================================================================== #
#|                                helper functions                                    |#
# ==================================================================================== #


# ==================================================================================== #
#|                                   main tests                                       |#
# ==================================================================================== #



def _observe_saved_data():
    file_name =  "learned_results 2022.10.08_12.35.11"
    results : LearnedResults = saveload.load(file_name)
    ## Unpack results:
    theta = results.theta
    final_state = results.state
    similarity = results.similarity
    ## Derive:
    num_moments = final_state.shape[0]-1
    initial_state = Fock(0).to_density_matrix(num_moments=num_moments)
    ## Movie:
    coherent_control = CoherentControl(num_moments=num_moments)
    coherent_control.coherent_sequence(initial_state, theta, record_video=True)
    ## Plot:
    # draw_now()
    # state_plot = visuals.MatterStatePlot()
    # state_plot.update(final_state, title=f"similarity = {similarity}")
    ## Done:
    print("Done")
    
    

def main():
    _observe_saved_data()
    


if __name__ == "__main__":
    main()

