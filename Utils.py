import time 
import typing as typ
from abc import ABC
from collections.abc import Iterable


class Decorators(ABC):

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
    def assertType(Type: type, atInput: bool = True, atOutput: bool = True) -> typ.Callable:
        def decorator(func: typ.Callable):
            def wrapper(*args, **kwargs):
                
                def _Assertion(arg):
                    assert isinstance(arg, Type), f"Assertion failed. Argument {arg} is not of type '{Type.__name__}'"

                # assert input :
                if atInput:
                    for arg in args:
                        _Assertion(arg)
                    for key, val in kwargs.items():
                        _Assertion(val)

                # Call func:
                results = func(*args, **kwargs)

                # assert outputs:
                if atOutput:
                    if isinstance(results, Iterable):
                        for result in results:
                            _Assertion(result)
                    else:
                        _Assertion(results)

                return results           
            return wrapper
        return decorator


@Decorators.assertType(int, atInput=False, atOutput=True)
@Decorators.assertType(str, atInput=True , atOutput=False)
def _ExampleFunc(a, b, c='Hello', d='Bye'):    
    print(f"a='{a}' , b='{b}' , c='{c}' , d='{d}'")
    res = 3
    return res

if __name__ == "__main__":
    # Run Example Code
    _ExampleFunc(3, 'Gutman', d="Cio")
