import functools
import numpy as np
from typing import Self, NamedTuple, Literal, Final

from .fock import FockState
from ._base import StatesSum, _StateBase, Operator
from .qubits import QubitsState, QubitsSum, QubitOperator
from ...utils import assertions, tuples


DEBUG : Final[bool] = True

class DickeState(_StateBase["DickeSum"]):
    __slots__ = ("excitations", "num_qubits")
    """           Creation:        """
    """  ========================= """

    def __init__(self, excitations:int, num_qubits:int) -> None:
        super().__init__()
        self.excitations : int = assertions.index(excitations, reason=f"Excitations must be a positive integer. Not {excitations!r}")
        self.num_qubits : int = assertions.index(num_qubits, reason=f"Number of qubits must be a positive integer. Not {num_qubits!r}")


    """  Abstract methods that must be implemented: """
    """  ========================================== """
    def copy(self):
        new = super().copy()
        new.excitations = self.excitations
        new.num_qubits = self.num_qubits
        return new
    
    def validate(self)->None: 
        assertions.index(self.excitations, reason=f"Dicke number must be 0 or a positive integer. Not {self.excitations!r}")
        assertions.index(self.num_qubits, reason=f"Dicke states has a positive number of qubits. Not {self.num_qubits!r}")
        assert self.excitations <= self.num_qubits, "Dicke state cannot have more excitations than qubits."

    def _inner_state_str(self)->str: 
        return f"{self.excitations}"

    def _is_same_state(self, other:Self) -> bool: 
        return self.excitations == other.excitations \
            and self.num_qubits == other.num_qubits
    
    def _order_lambda(self) -> int:
        return self.excitations       


    """     self methods and properties:   """
    """  ================================= """
    def to_qubits(self) -> QubitsSum:
        qubits_list = _symmetric_state_to_qubit_state(num_qubits=self.num_qubits, excitation=self.excitations)
        qubits_sum = QubitsSum()
        total_count = 0
        for item in qubits_list:
            if DEBUG:
                assert sum(item.bits)==self.excitations, f"Sum of bits must be equal to the excitation. Not {item.bits} != {self.excitations}"
            qubits_state = QubitsState(*item.bits) * item.count
            total_count += item.count
            qubits_sum += qubits_state
        qubits_sum.normalize()
        qubits_sum *= self.weight
        return qubits_sum
    
    """           Visualizations           """
    """  ================================= """

    def __repr__(self) -> str:
        s = super().__repr__()
        s += f"_{self.num_qubits}"
        return s

    def plot_block_sphere(self) -> None:
        raise NotImplementedError("block sphere visualization not implemented yet.")
    


class DickeSum(StatesSum[FockState]):

    def _class_name_str(self) -> str:
        if len(self)==0:
            return "Dicke-Sum"
        else:
            n = self[0].num_qubits
            for state in self:
                assert state.num_qubits == n, "All states in the sum must have the same number of qubits."
            return f"Dicke-Sum({n})"



class DickeBasis:
    """ factory design pattern for creating Dicke states. """
    def __init__(self, num_qubits:int) -> None:
        self.num_qubits : int = assertions.index(num_qubits, reason=f"Number of qubits must be a positive integer. Not {num_qubits!r}")

    def __call__(self, excitation:int) -> DickeState:
        return DickeState(excitation, self.num_qubits)


# ==================================================================================== #
# |                               Inner Functions and objects                         | #
# ==================================================================================== #

class _BitsAndCount(NamedTuple):
    bits : tuple[int] 
    count : int

class _SetOfBitsAndCounts():
    def __init__(self):
        self.full_list : list[_BitsAndCount] = list()
        self.bits_set : set[tuple[int]] = set()
    
    def __contains__(self, item:tuple[int]) -> bool:
        return item in self.bits_set
    
    def _search_slot(self, item:tuple[int]) -> int:
        for i, bits_and_count in enumerate(self.full_list):
            if bits_and_count.bits == item:
                return i
        raise ValueError(f"Item {item} not found in the list.")

    def add(self, item:tuple[int]) -> None:
        assert item not in self
        self.bits_set.add(item)
        self.full_list.append(_BitsAndCount(item, 1))

    def increment(self, item:tuple[int]) -> None:
        assert item in self
        i = self._search_slot(item)
        prev_count = self.full_list[i].count + 1
        self.full_list[i] = _BitsAndCount(item, prev_count+1)

    def to_list(self) -> list[_BitsAndCount]:
        return self.full_list

    def __repr__(self) -> str:
        s = f"{self.__class__.__name__}:"
        for item in self.full_list:
            s += f"\n    {item}"
        return s


def _generate_permutations_with_higher_energy(state:tuple[int], num_qubits:int)->list[tuple[int]]:
    permutations = []
    for i in range(num_qubits):
        if state[i]==1:
            continue
        new_state = tuples.copy_with_replaced_val_at_index(state, i, 1)
        permutations.append(new_state)
    return permutations

    
@functools.cache
def _symmetric_state_to_qubit_state(num_qubits:int, excitation:int)->list[_BitsAndCount]:
    if excitation==0:
        return [ _BitsAndCount(tuple([0]*num_qubits), 1) ]
    lower_excitation_symmetric_state = _symmetric_state_to_qubit_state(num_qubits=num_qubits, excitation=excitation-1)
    qubits_and_counts : _SetOfBitsAndCounts = _SetOfBitsAndCounts()
    for lower_state, count in lower_excitation_symmetric_state:
        permutations = _generate_permutations_with_higher_energy(lower_state, num_qubits)
        for permutation in permutations:  # look-up by the `bits` part only
            if permutation in qubits_and_counts:
                qubits_and_counts.increment(permutation)
            else:
                qubits_and_counts.add(permutation)
    return qubits_and_counts.to_list()
    


# ==================================================================================== #
# |                               Operators                                          | #
# ==================================================================================== #
class S(Operator[DickeState]):
    def __init__(self, sign:str) -> None:
        self.sign : str = sign

    def _apply_on_ket_no_weight(self, state:DickeState) -> DickeState:
        # Parse common variables:
        m = state.excitations
        n = state.num_qubits
        # S+|e,n> = sqrt((e+1)(n-e)) |e+1,n>
        if self.sign == "+":  
            if m == n:
                return 0*DickeState(0, n)
            new_state = DickeState(m+1, n)
            weight = np.sqrt((m+1)*(n-m))
        # S-|e,n> = sqrt(e(n-e+1)) |e-1,n>
        elif self.sign == "-":
            if m == 0:
                return 0*DickeState(0, n)
            new_state = DickeState(m-1, n)
            weight = np.sqrt(m*(n-m+1))
        # Sz|e,n> = (e-n/2) |e,n>
        elif self.sign == "z":
            new_state = DickeState(m, n)
            weight = m - (n/2)
        else:
            raise ValueError(f"Sign {self.sign} not recognized.")
        return new_state * weight
    
    def _sign_str(self) -> str:
        return self.sign

    def __repr__(self) -> str:
        return f"S{self._sign_str()}"
    

Sp = S("+")
Sm = S("-")
Sz = S("z")
Sx = Sp + Sm
Sy = 1j*(Sm - Sp)