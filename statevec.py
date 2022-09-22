from dataclasses import dataclass
import typing as typ
import numpy as np


from utils import (
    errors,
    assertions,
)

# For type hintings:
from typing import (
    Any,
    Literal,
    Optional,
)

from math import factorial

# Global Constants:
EPSILON = 0.00000001

# For typing hints:
_Qubit = typ.Literal[0,1]
_Qubits = typ.List[_Qubit]
_Ket = typ.TypeVar('_Ket')
T = typ.TypeVar('T')



def with_indicate_first(it: typ.Iterator[T] ) ->  typ.Generator[ typ.Tuple[bool, T], None, None ]:
    isFirst: bool = True
    for val in it:
        yield isFirst, val
        isFirst = False

@dataclass
class Expression():
    weight: complex
    qubits: _Qubits

    def is_zero_weight(self)->bool:
        return self.weight < EPSILON

class Ket():
    def __init__(self, *args) -> None:
        weight = 1.00
        qubits = list(args)
        self._n : int = len(qubits)
        self._expressions : typ.List[Expression] = list()
        self._expressions.append( Expression(weight=weight, qubits=qubits) )

    @property
    def is_normalized(self) -> bool:        
        return abs(self.norm_squared()-1)<EPSILON

    @property
    def num_qubits(self) -> int:
        return self._n

    @property
    def expressions(self) -> typ.Generator[Expression, None, None]:
        for expression in self._expressions:
            yield expression

    def norm_squared(self) -> float:
        sum_squared = 0.00
        for ket_exp in self._expressions:
            for bra_exp in self._expressions:
                overlap = 1.00
                for i in range(self._n):
                    ket_qubit = ket_exp.qubits[i]
                    bra_qubit = bra_exp.qubits[i]
                    overlap *= int(ket_qubit==bra_qubit)
                sum_squared += overlap*ket_exp.weight*np.conj(bra_exp.weight)
        return sum_squared

    def norm(self) -> float:
        return np.sqrt(self.norm_squared())

    def _validate_output(func: typ.Callable) -> typ.Callable:
        def wrapper(*args):
            result = func(*args)
            assert result.is_normalized
            return result
        return wrapper

    def _validate_operator_inputs(func: typ.Callable) -> typ.Callable:
        def wrapper(*args):
            assert args[0]._n == args[1]._n
            results = func(*args)
            return results
        return wrapper
    
    def _drop_zero_weights(self)->None:
        self._expressions[:] = [ exp for exp in self._expressions if not exp.is_zero_weight() ]


    @_validate_operator_inputs
    def __add__(self, other: _Ket) -> _Ket:
        for expression in other._expressions:
            self._expressions.append( expression )            
        self._drop_zero_weights()
        return self

    def __sub__(self, other: _Ket) -> _Ket:
        return self.__add__(other*(-1))

    def __mul__(self, other: complex) -> _Ket:
        assert isinstance(other, (complex, float, int))
        for expression in self._expressions:
            expression.weight = expression.weight * other        
        return self

    def __rmul__(self, other: float) -> _Ket:
        return self.__mul__(other)
    
    def __repr__(self) -> str:
        string = ""
        for is_first_exp, expression in with_indicate_first(self._expressions):
            expression_str = f"({expression.weight})|"
            for is_first_qu, qubit in with_indicate_first(expression.qubits):
                qubit_str = f"{qubit}"
                if not is_first_qu:
                    qubit_str = ''+qubit_str
                expression_str += qubit_str
            expression_str += f">"
            # concatenate:
            if not is_first_exp:
                expression_str = ' + '+expression_str
            string += expression_str
        return string

    def __and__(self, other: _Ket) -> _Ket:
        pass


def _dec2binary(n: int, length: Optional[int] = None ): 
    # Transform integer to binary string:
    if length is None:
        binary_str = f"{n:0b}" 
    else:
        binary_str = f"{n:0{length}b}" 
    # Transform binary string to list:
    binary_list = [int(c) for c in binary_str]  
    return binary_list

class Fock(Ket):
    def __init__(self, n:int, num_bits:Optional[int]=None) -> None:
        bits = _dec2binary(n, length=num_bits)
        super().__init__(bits)

class FockSpace():
    def __init__(self, max_num:int) -> None:
        assertions.index(max_num)
        self.max_num = max_num

    @property
    def num_bits(self) -> int:
        bits = _dec2binary(self.max_num)
        return len(bits)
    @num_bits.setter
    def num_bits(self, val:Any) -> None:
        raise errors.ProtectedPropertyError("Can't change property `num_bits`")
    
    def state(self, n:int) -> Fock:
        return Fock(n, num_bits=self.num_bits)



def coherent_state(max_num:int, alpha:float, type_:Literal['normal', 'even_cat', 'odd_cat']='normal')->Fock:
    # Choose iterator:
    if type_=='normal':
        iterator = range(0, max_num+1, 1)
    elif type_=='even_cat':
        iterator = range(0, max_num+1, 2)
    elif type_=='odd_cat':
        iterator = range(1, max_num+1, 2)
    else:
        raise ValueError("`type_` must be either 'normal', 'even_cat' or 'odd_cat'.")

    # fock space object:
    fock_space = FockSpace(max_num)

    # Iterate
    first = True
    for n in iterator:
        # choose coeficient
        coef = np.exp( -(abs(alpha)**2)/2 ) * np.power(alpha, n) / np.sqrt( factorial(n) )
        # Create ket state:
        if first:
            ket : Ket = fock_space.state(n) * coef
        else:
            ket += fock_space.state(n) * coef
        # Update:
        first = False
    
    # Normalize:
    ket *= 1/( ket.norm() )
    return ket

def _main_test():
    k0 = Ket(0,0)
    k1 = Ket(1,1)
    k : Ket = (k0+k1)*(1/np.sqrt(2))
    print(f"is_normalized = '{k.is_normalized}'")
    print(k)

def _fock_test():
    fock_space = FockSpace(4)

    f0 = fock_space.state(0)
    f2 = fock_space.state(2)
    f = ( f0 + f2 )*(1/np.sqrt(2))
    print(f"is_normalized = '{f.is_normalized}'")
    print(f)

def _cat_state_test():
    zero = coherent_state(4, 0.0, 'normal')
    print(zero)
    cat = coherent_state(8, 1.0, 'even_cat')
    print(cat)
    print(f"norm squared = {cat.norm_squared()}")

if __name__ == "__main__":
    _cat_state_test()
    _fock_test()
    _main_test()