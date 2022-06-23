import time 
import typing as typ
from collections.abc import Iterable

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

if __name__ == "__main__":
    # Run Example Codes:
    _assertType_Example('3', 'Gutman', d="Cio")

