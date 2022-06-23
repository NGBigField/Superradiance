from typing import TypeVar, Generic

# Base:
class _UnitsBase:
    @classmethod
    def __str__(cls) -> str:
        pass

class Units: 
    # Basic 
    class UnitLess(_UnitsBase): ...
    class Undefined(_UnitsBase): ...

    # Common:
    class Radians(_UnitsBase):
        @classmethod
        def __str__(cls) -> str:
            return "Rad"
    Rad = Radians  # Alias

    class Seconds(_UnitsBase): 
        @classmethod
        def __str__(cls) -> str:
            return "sec"
    Sec = Seconds  # Alias

    class Kilograms(_UnitsBase):
        @classmethod
        def __str__(cls) -> str:
            return "Kg"
    Kg = Kilograms  # Alias


_InputType = TypeVar('_InputType')

class Variable(Generic[_InputType]):
    """Variable a number that is associated with some units and a optional scale.

    Args:
        Generic (_type_): can be int, float or complex.
    """
    def __init__(self, value:_InputType, units:_UnitsBase=Units.Undefined, scale:float=1.0) -> None:
        self.value = value
        self.units = units
        self.scale = scale

    def __repr__(self) -> str:
        newline = "\n"
        str = "Variable with:"+newline
        str += f"value: {self.value}"+newline
        str += f"type : {type(self.value)}"+newline
        str += f"units: {self.units.__str__()}"+newline
        str += f"scale: {self.scale}"
        return str

    def __str__(self) -> str:
        unit_str = f" [{self.units.__str__()}]"
        if self.scale == 1.0:
            scale_str = ""
        else:
            scale_str = f" x({self.scale})"
        value_str = f"{self.value}"
        str = value_str+scale_str+unit_str
        return str

if __name__ == "__main__":
    x = Variable[float](3.4, Units.Kg)
    print(x)
    y = Variable[float](0.1, Units.Kg)
    z = x + y
    print(z)