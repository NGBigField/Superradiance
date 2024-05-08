
from typing import TypeAlias

RealValue : TypeAlias = float|int






def linear_interpolation_by_range(x:RealValue, x_range:tuple[RealValue, RealValue], y_range:tuple[RealValue, RealValue])->RealValue:
    """_summary_

    Args:
        x (float)
        x_range (tuple[float, float])
        y_range (tuple[float, float])

    Returns:
        float: y. Value of
    """

    ## Simplest cases first:
    if x <= x_range[0]:
        return y_range[0]
    
    if x >= x_range[1]:
        return y_range[1]
    
    ## Linear function 
    m = (y_range[1] - y_range[0]) / (x_range[1] - x_range[0])
    n = y_range[1] - m*x_range[1]   # Find intersection with y axis
    y = m*x + n

    return y