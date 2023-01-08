from typing import (
    List,
    Any,
    Generator,
    TypeVar,
    Tuple,
)

_T = TypeVar('_T')



def same_length(*args:List[Any]) -> bool:
    is_first = True
    common_length : int = -1
    for lis in args:
        assert isinstance(lis, list), f"All inputs must be lists, but got input of type {type(lis)}"
        if is_first:
            common_length = len(lis)
            is_first = False
            continue
        if  len(lis)!=common_length:
            return False
    return True

def common_type(lis:List[Any]) -> type:
    is_first = True
    common_type : type = None
    for item in lis:
        if is_first:
            common_type = type(item)
            is_first = False
            continue
        
        crnt_type = type(item)
        assert crnt_type == common_type
    return common_type


def iterate_with_edge_indicators(l:List[_T]) -> Generator[Tuple[bool, bool, _T], None, None]:
    is_first : bool = True
    is_last  : bool = False
    n = len(l)
    for i, item in enumerate(l):
        if i+1==n:
            is_last = True
        
        yield is_first, is_last, item

        is_first = False