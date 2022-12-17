
# ==================================================================================== #
#|                                    Imports                                         |#
# ==================================================================================== #

import numpy as np

from typing import (
    Tuple,
)


# ==================================================================================== #
#|                                   Constants                                        |#
# ==================================================================================== #


# ==================================================================================== #
#|                                inner functions                                     |#
# ==================================================================================== #

def _type_from_num_or_type(num_or_type) -> type:
    if isinstance(num_or_type, type):
        return num_or_type
    else:
        return type(num_or_type)
# ==================================================================================== #
#|                               declared functions                                   |#
# ==================================================================================== #

def numpy_dtype_to_std_type(dtype:np.dtype) -> type :    
    numpy_variable = dtype.type(0)
    built_in_variable = numpy_variable.item()
    return type(built_in_variable)

def is_numpy_complex_type( num_or_type ) -> bool:
    # Get type:
    type_ = _type_from_num_or_type(num_or_type)
    # check inheritance:
    return issubclass(type_, np.complexfloating)
    

def is_numpy_float_type( num_or_type ) -> bool:
    # Get type:
    type_ = _type_from_num_or_type(num_or_type)
    # check inheritance:
    return issubclass(type_, np.floating)

def greatest_common_class(*types):
    mros = (type_.mro() for type_ in types)
    mro = next(mros)
    common = set(mro).intersection(*mros)
    return next((x for x in mro if x in common), None)

def greatest_common_numeric_class(*types: Tuple[type, ...]) -> type:
    # Define hierarchy:
    types_cascade = (int, float, complex)
    seen_types = [False, False, False]
    # Get info:
    for type_ in types:
        assert type_ in types_cascade, f"Can only check on numeric types. Given type was `{type_}` "
        index = types_cascade.index(type_)
        seen_types[index] = True
    # Deduce answer:
    common_type = types_cascade[ 
        max([i for i, seen in enumerate(seen_types) if seen]) 
    ]
    return common_type
    

    
    