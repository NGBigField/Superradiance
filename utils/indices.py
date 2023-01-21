# Typing hints:
from typing import (
    Tuple, 
    Iterator,
)

# For smart iterations:
import itertools


# ============================================================================ #
#                               Global Functions                               #
# ============================================================================ #

def all_possible_indices(dimensions: Tuple[int, ...]) -> Iterator :
    """iterate_permutation 

    Loop over all possible values of multiple indices within given dimensions

    Args:
        dimensions (typ.List[int]): The Shape of tensor for which we want to iterate over all possible indices.

    Returns:
        _type_: _description_
    """
    return itertools.product(*(range(dim) for dim in dimensions) )

