from typing import (
    List,
    Any,
    Generator,
    TypeVar,
    Tuple,
)
import numpy as np

_T = TypeVar('_T')


def add(l1:List[_T], l2:List[_T])->List[_T]:
    l3 = []
    for v1, v2 in zip(l1, l2, strict=True):
        l3.append(v1+v2)
    return l3

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

def common_super_class(lis:List[Any]) -> type:
    classes = [type(x).mro() for x in lis]
    for x in classes[0]:
        if all(x in mro for mro in classes):
            return x


def iterate_with_edge_indicators(l:List[_T]) -> Generator[Tuple[bool, bool, _T], None, None]:
    is_first : bool = True
    is_last  : bool = False
    n = len(l)
    for i, item in enumerate(l):
        if i+1==n:
            is_last = True
        
        yield is_first, is_last, item

        is_first = False

def random_item(lis:List[_T])->_T:
    n = len(lis)
    index = np.random.choice(n)
    return lis[index]