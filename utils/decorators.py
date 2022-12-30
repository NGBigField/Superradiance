from typing import (
    Literal,
    Callable,
    Iterable,
    Any,
)

# for sleeping between tries:
from time import sleep



def sparse_execution(skip_num:int, default_results:Any) -> Callable[[Callable], Callable]:
    assert isinstance(skip_num, int)
    assert skip_num > 0

    def decorator(func:Callable) -> Callable:
        counter : int = 0
    
        def wrapper(*args, **kwargs) -> Any:
            nonlocal counter
            
            if counter >= skip_num:
                results = func(*args, **kwargs)
                counter = 0
            else:
                results = default_results
                counter += 1
            
            return results
        return wrapper
    return decorator


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

def final_value(func: Callable) -> Callable :
    def wrapper(*args, **kwargs):
        # Parse Input:
        final_value_only = kwargs['final_value_only']
        # Run:
        results = func(*args, **kwargs)
        # return:
        if final_value_only:
            return results[-1]  
        else:
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


# = # = # = # = # = # = # = # = # = # = # = # = # = # = # = # = # = # = # = # = # = # = # = # = # = # = # = # = # = # = #

def save_results(name:str)->Callable[[Callable], Callable]: # function that returns a decorator
    def decorator(func:Callable)->Callable: # decorator that returns a wrapper:
        def wrapper(*args, **kwargs)->typ.Any: # wrapeer that cals the function            
            results = func(*args, **kwargs)
            Dict = dict(results=results, args=args, kwargs=kwargs, func_name=func.__name__)
            file_name = f"{name}_{Strings.time_stamp()}"
            SaveLoad.save(Dict, file_name)
            return results
        return wrapper
    return decorator

def multiple_tries(num:int, sleep_time_between_attempts:float=0.0)->Callable[[Callable], Callable]: # function that returns a decorator
    # Check input:
    assert sleep_time_between_attempts>=0.00, f"sleep_time_between_attempts must be a non-negative float."
    # Return decorator:
    def decorator(func:Callable)->Callable: # decorator that returns a wrapper:
        def wrapper(*args, **kwargs)->Any: # wrapeer that cals the function            
            func_name = func.__name__
            last_error = Exception("Temp Exception")
            for i in range(num):
                try:
                    results = func(*args, **kwargs)
                    return results
                except Exception as e:
                    last_error = e
                    print("")
                    print(f"Failed to run '{func_name}' at attempt {i+1}. Reason:")
                    print(str(e))
                    sleep(sleep_time_between_attempts)
            raise last_error
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


