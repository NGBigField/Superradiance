from typing import (
    Any,
    Optional,
    TypeVar,
    Union,
    Callable
)

_T = TypeVar('_T')

def default_value(
    arg:Union[None, _T], 
    default:_T=None, 
    default_factory:Optional[Callable[[], _T ]]=None
) -> _T :
    if arg is not None:
        return arg
    if default is not None:
        return default
    if default_factory is not None:
        return default_factory()
    raise ValueError(f"Must provide either `default` value or function `default_factory` that generates a value")
    
