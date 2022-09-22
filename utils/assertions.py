from typing import (
    Any,
)


def integer(x:Any)->Any:
    assert round(x) == x
    return x

def index(x:Any)->Any:
    integer(x)
    assert x>=0
    return x

def bit(x:Any)->Any:
    integer(x)
    assert x in [0, 1]
    return x

def even(x:Any)->Any:
    integer(x)
    assert float(x)/2 == int(int(x)/2)
    return x


