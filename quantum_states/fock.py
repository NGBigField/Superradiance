# ==================================================================================== #
# |                                   Imports                                        | #
# ==================================================================================== #
from __future__ import annotations

from dataclasses import dataclass, field

from typing import (
    List,
    Union,
)

from copy import deepcopy


# ==================================================================================== #
# |                                   Classes                                        | #
# ==================================================================================== #

@dataclass
class Fock():
    number : int 
    weight : complex = field(init=False, default=1.00)

    def similar(self, other:Fock) -> bool:
        assert isinstance(other, Fock)
        return self.number == other.number

    def __repr__(self) -> str:
        if self.weight == 1:
            return f"|{self.number}>"
        else:
            return f"{self.weight}|{self.number}>"

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

    def __mul__(self, other:float) -> Fock:
        assert isinstance(other, (float, int, complex) )
        self.weight *= other
        return self
        



@dataclass
class FockSum():
    states : List[Fock] = field(default_factory=list)

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

    def __len__(self) -> int:
        return len(self.states)

    def __getitem__(self, key) -> Fock:
        return self.states.__getitem__(key)

    def __setitem__(self, key, value:Fock) -> None:
        return self.states.__setitem__(key, value)

    def __delitem__(self, key) -> Fock:
        return self.states.__delitem__(key)

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
    f = -Fock(0)
    f *= 2
    print(f)
    fock = Fock(0) + Fock(0)*0.5 + Fock(3)*(1+1j*3) - Fock(1)
    print(fock)

if __name__ == "__main__":
    _test()