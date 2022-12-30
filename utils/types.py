
# ==================================================================================== #
#|                                    Imports                                         |#
# ==================================================================================== #

import numpy as np

from typing import (
    Tuple,
    Any,
    Dict,
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
    

def can_be_converted_to_dict(x:Any)->bool:
    if hasattr(x, "_asdict"):
        return True
    elif hasattr(x, "__dict__") and not isinstance(x, np.ndarray):
        return True
    else:
        return False

def as_plain_dict(x:Any) -> Dict[str, Any]:
    if hasattr(x, "_asdict"):
        d : Dict[str, Any] = x._asdict()
    elif hasattr(x, "__dict__"):
        d : Dict[str, Any] = x.__dict__
    else:
        raise TypeError(f"Input of type '{type(x)}' can't be converted to dict!")
    
    for key, val in d.items():
        if can_be_converted_to_dict(val):
            d[key] = as_plain_dict(val)
    return d