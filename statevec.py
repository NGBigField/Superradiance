from dataclasses import dataclass
import typing as typ
import numpy as np

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
    
    @_validate_operator_inputs
    def __add__(self, other: _Ket) -> _Ket:
        for expression in other._expressions:
            self._expressions.append( expression )
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




def main():
    k0 = Ket(0,0)
    k1 = Ket(1,1)
    k : Ket = (k0+k1)*(1/np.sqrt(2))
    print(f"is_normalized = '{k.is_normalized}'")
    print(k)

if __name__ == "__main__":
    main()