
import time 
import typing as typ

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
    return wrapper
