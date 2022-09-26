# ==================================================================================== #
# |                                   Imports                                        | #
# ==================================================================================== #
from __future__ import annotations

from utils import (
    assertions,
    errors,
    types,
)

from utils.errors import QuantumTheoryError


from dataclasses import dataclass, field, Field

from typing import (
    List,
    Union,
    Optional,
)

from copy import deepcopy

import numpy as np

from enum import Enum, auto

import warnings

# ==================================================================================== #
# |                                  Constants                                       | #
# ==================================================================================== #
EPS = 0.0000001  # for distance==0 checks


# ==================================================================================== #
# |                                    Errors                                        | #
# ==================================================================================== #
class InconsistentKetBraError(errors.QuantumTheoryError): ...

# ==================================================================================== #
# |                               Inner Functions                                    | #
# ==================================================================================== #
def _ket_or_bra_str( num:int, ket_or_bra:KetBra ) -> str:
    if ket_or_bra == KetBra.Ket:
        res = f"|{num}>"
    elif ket_or_bra == KetBra.Bra:
        res = f"<{num}|"
    else:
        raise Exception(f"Bug")
    return res


# ==================================================================================== #
# |                                    types                                         | #
# ==================================================================================== #
_NumericType = Union[int, float, complex]

# ==================================================================================== #
# |                                   Classes                                        | #
# ==================================================================================== #




class KetBra(Enum):
    Ket = auto()
    Bra = auto()

    def __invert__(self) -> KetBra :  # called in the operation ~self
        if self == KetBra.Ket:
            self = KetBra.Bra
        elif self == KetBra.Bra:
            self = KetBra.Ket
        else:
            raise Exception(f"Bug")
        return self

    def __repr__(self) -> str:
        return self.name



@dataclass
class Fock():
    number : int 
    weight : complex = field(init=False, default=1)
    ket_or_bra : KetBra = field(default=KetBra.Ket)

    def __post_init__(self):
        self.validate()

    def validate(self) -> None:
        assertions.index(self.number, f" fock-space-number must be an integer >= 0. Got `{self.number}`") 

    def similar(self, other:Fock) -> bool:
        assert isinstance(other, Fock)
        return self.number == other.number

    def to_sum(self) -> FockSum:
        sum = FockSum()
        sum += self
        return sum        

    def to_density_matrix(self, max_num:Optional[int]=None) -> np.matrix:
        return self.to_sum().to_density_matrix(max_num=max_num)

    @property
    def date_type(self) -> np.dtype:
        data_type = type(self.weight)
        if types.is_numpy_float_type(data_type):
            data_type = float
        return data_type

    @property
    def norm2(self) -> float:
        return abs(self.weight)**2 

    def __repr__(self) -> str:
        self.validate()
        ket_bra_str = _ket_or_bra_str(num=self.number, ket_or_bra=self.ket_or_bra)
        if self.weight == 1:
            return ket_bra_str
        else:
            return f"{self.weight}"+ket_bra_str

    def __neg__(self) -> Fock:
        self.weight *= -1
        return self

    def __add__(self, other:Fock) -> FockSum:
        assert isinstance(other, Fock)
        sum = FockSum()
        sum += self
        sum += other
        return sum

    def __sub__(self, other:Fock) -> FockSum:
        assert isinstance(other, Fock)
        return self + -other

    def __mul__(self, other:_NumericType) -> Fock:
        assert isinstance(other, (float, int, complex) )
        self.weight *= other
        return self

    def __rmul__(self, other:_NumericType) -> Fock:
        return self * other
        
    def __invert__(self) -> Fock :  # called in the operation ~self
        self.weight = np.conj( self.weight )
        self.ket_or_bra = ~self.ket_or_bra 



@dataclass
class FockSum():
    states : List[Fock] = field(default_factory=list, init=False)

    def to_density_matrix(self, max_num:Optional[int]=None) -> np.matrix:
        # Check input:
        if not self.normalized:
            warnings.warn("Create density matrices with normalized fock states")
        assert self.ket_or_bra == KetBra.Ket
        # Maximal fock number:
        if max_num is None:
            max_num = self.max_num
        # Prepare states:
        kets = self
        bras = ~self
        # Density size:
        n = max_num+1
        # Common type:
        common_data_type = self.date_type
        dtype = np.dtype(common_data_type)
        # Init output:
        mat = np.zeros((n,n))
        # Fill matrix:
        for ket in kets:
            for bra in bras:
                mat[ket.number, bra.number]  = ket.weight * bra.weight
        # Return matrix
        return mat
        
    @property
    def ket_or_bra(self) -> KetBra :
        kets_and_bras = [fock.ket_or_bra for fock in self] 
        if all( item==KetBra.Ket for item in kets_and_bras ):
            return KetBra.Ket
        elif all( item==KetBra.Bra for item in kets_and_bras ):
            return KetBra.Bra
        else:
            raise InconsistentKetBraError("Not all fock states are of same type")

    @property
    def norm2(self) -> float:
        res = 0.00
        for fock in self:
            res += fock.norm2
        return res

    @property
    def norm(self) -> float:
        return np.sqrt(self.norm2)

    @property
    def normalized(self) -> bool:
        return abs(self.norm2-1)<EPS

    @property
    def max_num(self) -> int:
        m = max(self.states, key=lambda fock: fock.number)
        return m.number

    @property
    def date_type(self) -> np.dtype:
        data_types = [fock.date_type for fock in self ]
        return types.greatest_common_numeric_class(*data_types) 

    def _add_state(self, fock:Fock) -> None:
        assert isinstance(fock, Fock)
        for i, inner_fock in enumerate(self.states):
            if inner_fock.similar(fock):
                self.states[i].weight += fock.weight
                break
        else:  # if none similar state was found
            self.states.append(fock)

    def _add_states(self, other:Union[Fock, FockSum], sign:int) -> None:
        assert sign in [-1, +1]
        
        if isinstance(other ,Fock):
            self._add_state(other*sign)
        elif isinstance(other ,FockSum):
            for fock in other:
                self._add_state(fock*sign)
        else:
            raise TypeError("Input `other` should be of type `Fock` or `FockSum`")
        return self
    
    def copy(self) -> FockSum :
        return deepcopy(self)

    def __len__(self) -> int:
        return len(self.states)

    def __getitem__(self, key) -> Fock:
        return self.states.__getitem__(key)

    def __setitem__(self, key, value:Fock) -> None:
        return self.states.__setitem__(key, value)

    def __delitem__(self, key) -> Fock:
        return self.states.__delitem__(key)

    def __mul__(self, other:Union[int, float, complex]) -> FockSum :
        assert isinstance(other, (int, float, complex))
        for fock in self:
            fock *= other
        return self

    def __truediv__(self, other:Union[int, float, complex]) -> FockSum :
        assert isinstance(other, (int, float, complex))
        self *= (1/other)
        return self

    def __invert__(self) -> FockSum:  # called in the operation ~self
        inverted = self.copy()
        for fock in inverted:
            ~fock
        return inverted

    def __add__(self, other:Union[Fock, FockSum]) -> FockSum:
        self += other
        return self

    def __sub__(self, other:Union[Fock, FockSum]) -> FockSum:
        self -= other
        return self

    def __iadd__(self, other:Union[Fock, FockSum]) -> FockSum:
        self._add_states(other, sign=+1)
        return self

    def __isub__(self, other:Union[Fock, FockSum]) -> FockSum:
        self._add_states(other, sign=-1)
        return self

    def __iter__(self):    
       return iter(self.states)

    def __repr__(self) -> str:
        states_sorted = deepcopy( sorted(self.states, key=lambda fock: fock.number ) )
        res = ""
        is_first : bool = True
        for fock in states_sorted:
            if is_first:
                res += str(fock)
                is_first = False
            else:
                weight = fock.weight
                if not isinstance(weight, complex) and weight<0:  # negative real number
                    connector = " - "
                    fock *= -1
                else:
                    connector = " + "
                res += connector + f"{fock}"            
        return res

# ==================================================================================== #
# |                                 main tests                                       | #
# ==================================================================================== #

def _test():

    fock : FockSum = Fock(0) - Fock(3)*np.sqrt(2) + (4+3j)*Fock(1)

    density_mat = fock.to_density_matrix()
    print(density_mat)

if __name__ == "__main__":
    _test()