from typing import (
    Literal,
    Callable,
    Iterable,
)

def timeit(func: Callable):
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

def assert_type(Type: type, at: Literal['input', 'output', 'both'] = 'both' ) -> Callable:
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
    def decorator(func: Callable):
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




# ============================================================================ #
#                                    Tests                                     #
# ============================================================================ #

@assert_type(int, at='output')
@assert_type(str, at='input' )
def _assert_type_example(a, b, c='Hello', d='Bye'):    
    print(f"a='{a}' , b='{b}' , c='{c}' , d='{d}'")
    res = 3
    print(f"res={res}")
    return res

def _test_assert_type():
    _assert_type_example('A', 'B', c="C")    
    

if __name__ == "__main__":
    _test_assert_type()

