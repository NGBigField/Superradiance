import time 
import typing as typ
from typing import Literal
from collections.abc import Iterable
from enum import Enum, auto

IS_FLAG = True 

class LiteralEnum(str, Enum):

    @classmethod
    def string_and_enums(cls) -> Literal:
        for i, (l, v) in enumerate(zip(cls.strings(), cls.__members__.values())):
            if i == 0:
                L = typ.Literal[l, v]
            else:
               L = typ.Union[L, typ.Literal[l, v]]
        return L

    @classmethod
    def all_literals(cls) -> Literal:
        for i, l in enumerate(cls.literals()):
            if i == 0:
                L = l
            else:
               L = typ.Union[L,l]
        return L

    @classmethod
    def literals(cls):        
        for s in cls.strings():            
            yield Literal[s]
    
    @classmethod
    def strings(cls) -> typ.Iterator[str]:
        for key in cls.__members__.keys():
            yield key



class InputOutput(LiteralEnum):
    Input  = auto()
    Output = auto()
    Both   = auto()

    def parse(self) -> typ.Tuple[bool, bool] :
        if self is InputOutput.Input:
            atInput  = True 
            atOutput = False
        elif self is InputOutput.Output:
            atInput  = False 
            atOutput = True
        elif self is InputOutput.Both:
            atInput  = True 
            atOutput = True
        else:
            raise ValueError(f"Impossible Option")
        return atInput, atOutput

L = InputOutput.all_literals()

def color(l: InputOutput.string_and_enums() ) -> int:
    return 32

a = color()

l = [ s for s in InputOutput.__members__.keys() ]
l

class TypeHints():

    @staticmethod
    def enumToLiteral( enumIn: Enum ) -> typ.Literal :
        
        strings : typ.List[str] = []
        values = []
        members = enumIn.__members__
        for key, val in enumIn.__members__.items():
            print(f"name={key} , val={val}")
            strings.append( key )
            values.append( val )

        L = typ.Literal[ 'x', 'y' ]
        return L


class Decorators():

    @staticmethod
    def timeit(func: typ.Callable):
        def wrapper(*args, **kwargs):
            # Parse Input:
            funcName = func.__name__
            # Run:
            startingTime = time.time()
            results = func(*args, **kwargs)
            finishTime = time.time()
            # Print:
            print(f"Function '{funcName}' took {finishTime-startingTime} sec.")
            # return:
            return results            
        return wrapper

    @staticmethod
    def assertType(Type: type, at: typ.Literal['input', 'output', 'both'] = 'both' ) -> typ.Callable:
        """assertType 

        Args:
            Type (type): type needed at input\output of function.
            at (InputOutput, optional): Defaults to InputOutput.Both.
        """
        # Parse options:
        if at == 'input':
            atInput  = True 
            atOutput = False
        elif at == 'output':
            atInput  = False 
            atOutput = True
        elif at == "both":
            atInput  = True 
            atOutput = True
        else:
            raise ValueError(f"Impossible Option")
        def decorator(func: typ.Callable):
            def wrapper(*args, **kwargs):
                # Define Assertion including error message:
                def _Assertion(arg):
                    assert isinstance(arg, Type), f"Assertion failed. Argument {arg} is not of type '{Type.__name__}'"
                # Assert Inputs :
                if atInput:
                    for arg in args:
                        _Assertion(arg)
                    for key, val in kwargs.items():
                        _Assertion(val)
                # Call func:
                results = func(*args, **kwargs)
                # Assert Outputs:
                if atOutput:
                    if isinstance(results, Iterable):
                        for result in results:
                            _Assertion(result)
                    else:
                        _Assertion(results)
                return results           
            return wrapper
        return decorator


@Decorators.assertType(int, at='output')
@Decorators.assertType(str, at='input' )
def _assertType_Example(a, b, c='Hello', d='Bye'):    
    print(f"a='{a}' , b='{b}' , c='{c}' , d='{d}'")
    res = 3
    print(f"res={res}")
    return res

InputOutputArg : typ.Literal = TypeHints.enumToLiteral( InputOutput )
def _enumHint_Example( at: InputOutputArg):
    print(at)


if __name__ == "__main__":
    # Run Example Codes:
    # _assertType_Example('3', 'Gutman', d="Cio")
    _enumHint_Example()
