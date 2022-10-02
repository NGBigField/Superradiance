from typing import (
    Any,
    Optional,
    TypeVar,
    Union,
)

_T = TypeVar('_T')

def default_value(arg:Union[None, _T], default:_T) -> _T :
    if arg is None:
        return default
    else:
        return arg