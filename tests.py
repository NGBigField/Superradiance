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
from fock import Fock

# For coherent control
from coherentcontrol import CoherentControl

# for plotting
import matplotlib.pyplot as plt  # for plotting test results:

# For optimizations:
from optimization import learn_specific_state, LearnedResults, _coherent_control_from_mat

# ==================================================================================== #
#|                                helper functions                                    |#
# ==================================================================================== #


# ==================================================================================== #
#|                                   main tests                                       |#
# ==================================================================================== #



def _observe_saved_data():
    file_name =  "learned_results 2022.11.15_18.34.18"
    results : LearnedResults = saveload.load(file_name)
    ## Unpack results:
    theta           = results.theta
    initial_state   = results.initial_state
    final_state     = results.final_state
    score           = results.score
    print(results)
    ## Plot:
    visuals.plot_matter_state(final_state, num_points=150)
    ## Movie:
    # coherent_control = _coherent_control_from_mat(final_state)
    # coherent_control.coherent_sequence(initial_state, theta, movie_config=CoherentControl.MovieConfig(
    #     active=True,
    #     show_now=False,
    #     num_transition_frames=3,        
    # ))
    ## Done:
    print("Done")
    
    

def main():
    _observe_saved_data()
    


if __name__ == "__main__":
    np_utils.fix_print_length()
    main()

