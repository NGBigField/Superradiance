# ==================================================================================== #
# |                                   Imports                                        | #
# ==================================================================================== #

from typing import (
    Any,
    Optional,
)


# ==================================================================================== #
# |                               Inner Functions                                    | #
# ==================================================================================== #

def _assert(is_true:bool, reason:Optional[str]=None):
    if is_true:
        return
    if reason is not None and isinstance(reason, str):
        raise AssertionError(reason)
    else:
        raise AssertionError()


# ==================================================================================== #
# |                              Declared Functions                                  | #
# ==================================================================================== #

def integer(x:Any, reason:Optional[str]=None) -> Any:
    _assert( round(x) == x, reason=reason )
    return x

def index(x:Any, reason:Optional[str]=None) -> Any:
    integer(x)
    _assert( x >= 0, reason=reason )
    return x

def bit(x:Any, reason:Optional[str]=None) -> Any:
    integer(x)
    _assert( x in [0, 1], reason=reason )
    return x

def even(x:Any, reason:Optional[str]=None) -> Any:
    integer(x)
    _assert( float(x)/2 == int(int(x)/2), reason=reason )
    return x


