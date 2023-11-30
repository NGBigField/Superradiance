# ==================================================================================== #
# |                                   Imports                                        | #
# ==================================================================================== #
from __future__ import annotations

from typing import (
    List,
    Union,
    Optional,
    Literal,
    overload,
)

# For relative import when needed:
import sys, pathlib
sys.path.append(f"{pathlib.Path(__file__).parent.parent}")

# Import our tools from another directory:
from utils import (
    assertions,
    errors,
    types,
    visuals,
    arguments,
    numpy_tools,
)


from dataclasses import dataclass, field


from copy import deepcopy

import numpy as np
import math

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


def _root_factorial(num:int) -> float :
    res = 1.0
    
    if num==0 or num==1:
        return res
    
    for i in range(1, num+1):
        res *= np.sqrt(i)

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
    
    @staticmethod
    def ground_state_density_matrix(num_atoms:int) -> np.matrix:
        return Fock(0).to_density_matrix(num_moments=num_atoms)
        
    @staticmethod
    def excited_state_density_matrix(num_atoms:int) -> np.matrix:
        state = Fock(0).to_density_matrix(num_moments=num_atoms)
        state[0,0] = 0
        state[-1,-1] = 1
        return state

    @overload
    @staticmethod
    def create_coherent_state(num_atoms:int, alpha:float, output:Literal['density_matrix'], type_:Literal['normal', 'even_cat', 'odd_cat']='normal')->np.matrix: ...
    @overload
    @staticmethod
    def create_coherent_state(num_atoms:int, alpha:float, output:Literal['ket'], type_:Literal['normal', 'even_cat', 'odd_cat']='normal')->FockSum: ...

    @staticmethod
    def create_coherent_state(
        num_atoms:int, 
        alpha:float, 
        output:Literal['ket', 'density_matrix']='ket',
        type_:Literal['normal', 'even_cat', 'odd_cat']='normal',
    )->Union[FockSum, np.matrix] :  
        ket = coherent_state(num_moments=num_atoms, alpha=alpha, type_=type_)
        if output == 'ket':
            return ket
        elif output == 'density_matrix':
            return ket.to_density_matrix(num_moments=num_atoms)
        else:
            raise ValueError("Not an option.")

    def validate(self) -> None:
        assertions.index(self.number, f" fock-space-number must be an integer >= 0. Got `{self.number}`") 

    def similar(self, other:Fock) -> bool:
        assert isinstance(other, Fock)
        return self.number == other.number

    def to_sum(self) -> FockSum:
        sum = FockSum()
        sum += self
        return sum        

    def to_density_matrix(self, num_moments:Optional[int]=None) -> np.matrix:
        return self.to_sum().to_density_matrix(num_moments=num_moments)

    def zero_weight(self) -> bool:
        return abs(self.weight)<EPS

    @property
    def date_type(self) -> np.dtype:
        data_type = type(self.weight)
        if types.is_numpy_float_type(data_type):
            return float
        if types.is_numpy_complex_type(data_type):
            return complex
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

    def to_density_matrix(self, num_moments:Optional[int]=None) -> np.matrix:
        # Check input:
        if not self.normalized:
            warnings.warn("Create density matrices with normalized fock states")
        assert self.ket_or_bra == KetBra.Ket
        # Maximal fock number:
        num_moments = arguments.default_value(num_moments, self.max_num)
        # Prepare states:
        kets = self
        bras = ~self
        # Density size:
        n = num_moments+1
        # Common type:
        common_data_type = self.date_type
        dtype = np.dtype(common_data_type)
        # Init output:
        mat = np.zeros((n,n), dtype=dtype)
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
        if fock.zero_weight():
            return  # Avoid adding states with zero contribution 
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
# |                             Declared Functions                                   | #
# ==================================================================================== #

def cat_state(num_atoms:int, alpha:float, num_legs:int, phase:float=0.0)->FockSum:
    """cat_state _summary_

    _extended_summary_

    Args:
        num_atoms (int): 
        alpha (float): 
        num_legs (int): 
        phase (float, optional):  Defaults to 0.0.
            if phase == 0  then we get an even cat.
            if phase == pi then we get an odd  cat.
            else, some other relative phase between each macroscopic state.
            

    Returns:
        FockSum: 
    """
    
    # check inputs:
    num_atoms = assertions.integer(num_atoms)
    num_legs = assertions.integer(num_legs)

    # fock space object:
    fock = FockSum()

    # on moments
    for moment in range(0, num_atoms+1):
        
        total_coefficient = 0.0
        for leg_index in range(num_legs):
            leg_alpha = alpha * np.exp(1j * 2*np.pi * leg_index / num_legs)
            leg_alpha = numpy_tools.reduce_small_imaginary_to_zero(leg_alpha)

            # compute coefficient
            if leg_alpha == 0:
                coef = 0
            else:
                power = np.power(leg_alpha, moment) 
                coef = np.exp( -(abs(leg_alpha)**2)/2 ) * power / _root_factorial(moment) 
                
            if phase!=0.0:
                exponent = 1j*phase * leg_index 
                coef *= numpy_tools.reduce_small_imaginary_to_zero( np.exp(exponent) )
                
            total_coefficient += coef

        fock += Fock(moment)*total_coefficient
    
    # Normalize:
    fock /= fock.norm
    return fock


def coherent_state(num_moments:int, alpha:float, type_:Literal['normal', 'even_cat', 'odd_cat']='normal')->FockSum:
    # Choose iterator:
    if type_=='normal':
        iterator = range(0, num_moments+1, 1)
    elif type_=='even_cat':
        iterator = range(0, num_moments+1, 2)
    elif type_=='odd_cat':
        iterator = range(1, num_moments+1, 2)
    else:
        raise ValueError("`type_` must be either 'normal', 'even_cat' or 'odd_cat'.")

    # fock space object:
    fock = FockSum()

    # Iterate
    for n in iterator:
        # choose coeficient
        power = np.power(alpha, n) 
        if power == 0:
            coef = 0
        else:
            coef = np.exp( -(abs(alpha)**2)/2 ) * power / np.sqrt( math.factorial(n) )
        fock += Fock(n)*coef
    
    # Normalize:
    fock /= fock.norm
    return fock

# ==================================================================================== #
# |                                 main tests                                       | #
# ==================================================================================== #

def _test_simple_fock_sum():

    fock : FockSum = Fock(0) - Fock(3)*2 
    fock /= fock.norm

    density_mat = fock.to_density_matrix()
    print(density_mat)
    print(f"trace={density_mat.trace()}")


def _test_coherent_state(max_fock_num:int=4):
    assertions.even(max_fock_num)

    zero = coherent_state(num_moments=4, alpha=0.00, type_='normal')
    print(zero)
    visuals.plot_city(zero.to_density_matrix(num_moments=max_fock_num))

    ket = coherent_state(num_moments=4, alpha=1.00, type_='normal')
    print(ket)
    visuals.plot_city(ket.to_density_matrix(num_moments=max_fock_num))

    print("Plotted.")

def _test_simple_fock_density_matrix():
    rho = Fock.create_coherent_state(num_atoms=2, alpha=0, output='density_matrix')
    print(rho)

def _test_cat_state(
    num_moments:int = 20,
    alpha:float = 3,
    num_legs:int = 2
):
    fock_sum = cat_state(num_atoms=num_moments, alpha=alpha, num_legs=num_legs, phase=np.pi/2)
    print(fock_sum)
    rho = fock_sum.to_density_matrix(num_moments=num_moments)
    # visuals.plot_matter_state(rho)
    visuals.plot_plain_wigner(rho)
    visuals.draw_now()
    print("Printed")

def _test_root_factorial(N:int=40):
    for n in range(N):
        print(f" == n={n:3} == ")
        try:
            fact1 = np.sqrt( math.factorial(n) )
        except:
            fact1 = None
        fact2 = _root_factorial(n)
        print(fact1)
        print(fact2)

if __name__ == "__main__":
    # _test_coherent_state()
    # _test_simple_fock_density_matrix()
    # _test_root_factorial()
    _test_cat_state()
    print("Done.")