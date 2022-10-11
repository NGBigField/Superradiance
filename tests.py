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
    file_name =  "learned_results 2022.10.11_14.10.14"
    results : LearnedResults = saveload.load(file_name)
    ## Unpack results:
    theta           = results.theta
    initial_state   = results.initial_state
    final_state     = results.final_state
    similarity      = results.similarity
    print(results)
    ## Derive:
    num_moments = final_state.shape[0]-1
    ## Movie:
    coherent_control = CoherentControl(num_moments=num_moments)
    coherent_control.coherent_sequence(initial_state, theta, movie_config=CoherentControl.MovieConfig(
        active=True,
        show_now=True
    ))
    ## Done:
    print("Done")
    
    

def main():
    _observe_saved_data()
    


if __name__ == "__main__":
    np_utils.fix_print_length()
    main()

