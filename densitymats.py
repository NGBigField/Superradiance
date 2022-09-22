# For progressive self type hinting:
from __future__ import annotations

import numpy as np
from numpy import typing as np_type
import typing as typ
import inspect
from copy import deepcopy
from statevec import Ket
from dataclasses import dataclass, field
from qiskit.quantum_info import random_density_matrix

from utils import (
    assertions,
)

EPSILON = 0.1

# For type hinting:
_DType = typ.TypeVar("_DType")
_BitType = typ.Literal[0, 1]
_BitsType = typ.List[_BitType]


def _is_static_or_instance_method(cls, method:typ.Callable) -> bool:
    # Filter info:
    attributes = inspect.classify_class_attrs(cls)
    matching = [a for a in attributes if a.name == method.__name__ ]
    attribute = matching[0]
    # Get type of method:
    is_static_method   = attribute.kind == 'static method'
    is_instance_method = attribute.kind == 'method'
        
    return is_static_method, is_instance_method

def _middle_index(mat: DensityMatrix) -> int:
    max_index = mat.dim
    middle_index = max_index//2
    return middle_index

def _ceil_of_half(x: int) -> int:
    """_ceil_of_half A quicker operation than int(np.ceil(x/2))
    """
    x = assertions.integer(x) 
    floored_half = x//2
    double_of_floored_half = floored_half*2
    if double_of_floored_half == x:
        return floored_half
    elif double_of_floored_half < x:
        return floored_half + 1
    else:
        raise ValueError(f"Error. Expected {floored_half}*2 <= {x} ")

def _half_the_qubits(mat: DensityMatrix) -> int:
    num_qubits = mat.num_qubits
    half_qubits = _ceil_of_half(num_qubits)
    return half_qubits

def _is_square(m: np.matrix) -> bool :
    shape = m.shape
    if len(shape) != 2:
        return False
    if shape[0] != shape[1]:
        return False
    return True

def _is_positive_semidefinite(m, EPSILON) -> bool:
    eigen_vals = np.linalg.eigvals(m)
    if np.any(np.imag(eigen_vals)>EPSILON):  # Must be real
        return False
    if np.any(np.real(eigen_vals)<-EPSILON):  # Must be positive
        return False
    return True

def _is_hermitian(m: np.matrix, tolerance=0.1) -> bool:
    diff = m.H-m
    shape = m.shape
    for i in range(shape[0]):
        for j in range(shape[1]):
            if abs(diff[i,j])>tolerance:
                return False
    return True    

T = typ.TypeVar('T')
def in_reverse(L: typ.List[T]) -> typ.Generator[T, None, None]:
    N = len(L)
    for i in range(N-1, -1, -1):
        yield L[i]

def _binary2dec(bits: _BitsType ) -> int:
    num = 0
    # Iterate over reversed order of bits
    for n, bit in enumerate(in_reverse(bits)):
        num += bit*(2**n)
    return num        

def _dec2binary(n: int, length: typ.Optional[int] = None ) -> _BitsType:  
    # Transform integer to binary string:
    if length is None:
        binary_str = f"{n:0b}" 
    else:
        binary_str = f"{n:0{length}b}" 
    # Transform binary string to list:
    binary_list = [int(c) for c in binary_str]  
    return binary_list

def _matrix_dimension_from_optional_num_qubits(num_qubits:typ.Optional[int] = None, mat_dim:typ.Optional[int] = None) -> int:
    if num_qubits is None and mat_dim is None:
        raise ValueError("Should get num_qubits or mat_length as input.")
    elif num_qubits is None and mat_dim is not None:
        pass
    elif num_qubits is not None and mat_dim is None:
        mat_dim = 2**num_qubits            
    elif num_qubits is not None and mat_dim is not None:
        if mat_dim != 2**num_qubits:
            raise ValueError("Mismatched arguments")
    return mat_dim

class SquareMatrix(np.matrix):
    error_msg = "Square-Matrix is not square"

    @property
    def dim(self) -> int:
        d = self.shape[0]
        assert d == self.shape[1], SquareMatrix.error_msg
        return d


    @staticmethod
    def empty(dim:int, dtype:typ.Optional[_DType]=None) -> SquareMatrix:
        if dtype is None:
            return SquareMatrix(np.empty((dim, dim)))
        else:
            return SquareMatrix(np.empty((dim, dim), dtype=dtype))

    @staticmethod
    def zeros(dim:int, dtype:typ.Optional[_DType]=None) -> SquareMatrix:
        if dtype is None:
            return SquareMatrix(np.zeros((dim, dim)))
        else:
            return SquareMatrix(np.zeros((dim, dim), dtype=dtype))

    def validate(self) -> None:
        assert _is_square(self), SquareMatrix.error_msg

    def __array_finalize__(self, obj: np_type.NDArray) -> None:
        if obj is None and self is not None:
            assert _is_square(self), SquareMatrix.error_msg
        elif obj is not None and self is None:
            assert _is_square(obj), SquareMatrix.error_msg
        elif obj is not None and self is not None:
            assert _is_square(obj), SquareMatrix.error_msg
            assert _is_square(self), SquareMatrix.error_msg
        else:
            raise ValueError("Expected self or obj to be of numpy.ndarray type")
        return super().__array_finalize__(obj)

    def all_indices(self) -> typ.Generator[ typ.Tuple[int, int], None, None ]:
        d = self.dim
        for i in range(d):
            for j in range(d):
                yield i, j 

    def to_numpy(self) -> np.matrix:
        shape = self.shape
        dtype = self.dtype
        out = np.zeros(shape=shape, dtype=dtype)
        for i, j in self.all_indices():
            out[i,j] = self[i,j]
        return out

class DensityMatrix(SquareMatrix):

    # Decorator for validation each step
    def validate_at(where:typ.Literal['before','after', 'results']):
        def decorator(method: typ.Callable) -> callable:
            def wrapper(*args, **kwargs):                
                # Check if user asked to validate or not:
                args_spec = inspect.getcallargs(method, *args, **kwargs)
                if 'validate' in args_spec:
                    on = args_spec['validate']
                else:
                    on = True


                # is static method?
                is_static_method, is_instance_method = _is_static_or_instance_method(DensityMatrix, method)                
                if is_instance_method:
                    self = args[0]

                # Validate where needed:
                if on and where=='before':
                    DensityMatrix.validate(args[0])            
                results = method(*args, **kwargs)   # <<<< The actual call
                if on and where=='results':
                    DensityMatrix.validate(results)
                if on and where=='after':
                    DensityMatrix.validate(args[0])            
                return results
            return wrapper
        return decorator

    @staticmethod
    def zeros(num_qubits:typ.Optional[int] = None, mat_dim:typ.Optional[int] = None, dtype:typ.Optional[_DType]=None) -> DensityMatrix:
        # Derive input:
        mat_dim = _matrix_dimension_from_optional_num_qubits(num_qubits, mat_dim)
        # Build matrix from super class:
        if dtype is None:
            m = DensityMatrix( np.zeros((mat_dim, mat_dim)) )
        else:
            m = DensityMatrix( np.zeros((mat_dim, mat_dim)), dtype=dtype )
        return m

    @staticmethod
    @validate_at('results')
    def random(
        num_qubits:typ.Optional[int] = None, 
        mat_dim:typ.Optional[int] = None, 
        method:typ.Literal["qiskit", "numpy"] = "qiskit",
        validate:bool = True
    ) -> DensityMatrix:
        # Derive input:
        mat_dim = _matrix_dimension_from_optional_num_qubits(num_qubits, mat_dim)
        # create matrix:
        if method == "qiskit":
            qiskit_mat = random_density_matrix(mat_dim)
            m = DensityMatrix( qiskit_mat.data )
        elif method == "numpy":
            rand = lambda dim: np.random.rand(dim, dim)
            m = rand(mat_dim) + 1j*rand(mat_dim)
            m *= 1/m.trace()
            m = DensityMatrix(m)
        return m

    @staticmethod
    @validate_at('results')
    def from_ket(ket: Ket) -> DensityMatrix:
        assert ket.is_normalized
        m = DensityMatrix.zeros(num_qubits=ket.num_qubits)
        for ket_exp in ket.expressions:
            ket_indices = ket_exp.qubits
            ket_weight  = ket_exp.weight
            for bra_exp in ket.expressions:
                bra_indices = bra_exp.qubits
                bra_weight  = np.conj(bra_exp.weight)
                i = DensityMatrix.Index(m, ket_indices)
                j = DensityMatrix.Index(m, bra_indices)
                weight = ket_weight*bra_weight
                m[i(),j()] = weight   
        return m


    def validate(self) -> None:
        assert _is_square(self), "Density Matrix must be a square matrix"  
        assertions.integer(np.log2(self.shape[0]))  # Must have indices of 2 to the power of the qubits
        assert abs(self.trace()-1)<EPSILON, "Density Matrix must have trace==1"
        assert _is_hermitian(self, EPSILON), "Density Matrix must be hermitian"
        assert _is_positive_semidefinite(self, EPSILON), "Density Matrix must be positive semidefinite"


    def trace(self) -> complex:
        return np.trace(self)

    @property
    def is_identity(self) -> bool:
        shape = self.shape
        for i in range(shape[0]):
            elem = self[i,i]
            if abs(elem - 1.00)>EPSILON:
                return False
        return True

    @property 
    def num_qubits(self) -> int:
        return assertions.integer( np.log2(self.dim) )

    def new_empty(self) -> DensityMatrix:
        new_ndarray = np.empty(self.shape)
        new_mat = DensityMatrix(new_ndarray)
        return new_mat

    def dagger(self) -> DensityMatrix:
        out = self.getH()
        return out

    @validate_at('before')
    def partial_transpose(self, num_qubits_on_first_part:typ.Optional[int]=None, part_to_transpose:typ.Literal['first', 'second']='first', validate:bool=True) -> _DensityMatrixType:
        return partial_transpose(self, num_qubits_on_first_part, part_to_transpose)
        
    @validate_at('before')
    def partial_trace(self):
        raise NotImplementedError
        

    class Index():
        """Index: 
            helper class for 'DensityMatrix' that makes indexing the matrix more convenient and simple.
        """
        def _set_values(self, index: int, binary: _BitsType) -> None:
            self.__index = index
            self.__binary = binary
        def __init__(self, mat: DensityMatrix, input: typ.Union[int, _BitsType]) -> None:
            self.length = mat.num_qubits
            if isinstance(input, int):
                input = assertions.index(input)
                self._set_values(index=input, binary=_dec2binary(input, length=self.length))
            elif isinstance(input, list):
                input = [assertions.bit(x) for x in input]
                self._set_values(index=_binary2dec(input), binary=input)
        @property
        def binary(self) -> _BitsType:
            return self.__binary
        @binary.setter
        def binary(self, bits: _BitsType) -> None:
            self._set_values(index=_binary2dec(bits), binary=bits)
        def __getitem__(self, key: int) -> _BitType:
            key = assertions.index(key)
            return self.binary[key]
        def __setitem__(self, key: int, val: _BitType) -> None:
            key = assertions.index(key)
            val = assertions.bit(val)
            binary = self.__binary
            binary[key] = val
            self._set_values(index=_binary2dec(binary), binary=binary)
        def __delitem__(self, key):
            raise NotImplemented
        @property
        def index(self) -> int:
            return self.__index
        @index.setter
        def index(self, ind: int):
            ind = assertions.index(ind)
            self._set_values(index=ind, binary=_dec2binary(ind, length=self.length))
        def __call__(self) -> int:
            return self.index
        def __repr__(self) -> str:
            return f"{self.binary}: {self.index}"
        # End DensityMatrix.Index    

class MatOfMats(SquareMatrix):

    @dataclass(init=False, eq=False, order=False)
    class Dims():
        Total : int = None 
        Super : int = None
        Sub   : int = None
        def validate(self) -> None:
            assert self.Sub*self.Super==self.Total, f"ValidationError. Expected: {self.Sub}*{self.Super}=={self.Total} "

    @staticmethod
    def empty(dim:int, dtype:typ.Optional[_DType]=None) -> MatOfMats:
        if dtype is None:
            return MatOfMats(np.empty((dim, dim)))
        else:
            return MatOfMats(np.empty((dim, dim), dtype=dtype))

    @staticmethod
    def from_mat(mat:DensityMatrix, num_qubits_on_first_part:int) -> MatOfMats :
        dims = MatOfMats.Dims()
        dims.Total = mat.dim
        dims.Super = 2**num_qubits_on_first_part
        dims.Sub = dims.Total//dims.Super
        dims.validate()
        super_mat = MatOfMats.empty(dims.Super, dtype=SquareMatrix)
        super_mat.dims = dims
        for i, j in super_mat.all_indices():
            super_mat[i,j] = MatOfMats._sub_mat(mat, dims.Sub, i, j)
        return super_mat

    @staticmethod
    def _sub_mat(mat, dim, i, j) -> np.matrix:
        _type = type(mat[0,0])
        sub_mat = SquareMatrix.zeros(dim, dtype=_type)    
        for k, l in sub_mat.all_indices():
            f = k + i*dim
            g = l + j*dim
            sub_mat[k, l] = mat[f, g]
        return sub_mat    

    def copy(self) -> MatOfMats:
        new = deepcopy(self)
        new._copy_attributes(self)
        return new

    def _copy_attributes(copy_to, copy_from: MatOfMats) -> None:
        copy_to.dims = copy_from.dims

    def __array_finalize__(self, obj ) -> None:
        self.dims = MatOfMats.Dims()
        return super().__array_finalize__(obj)

    def __repr__(self) -> str:
        s = super().__repr__()
        s += "\n"
        s += f"dims={self.dims}"
        return s

    def to_full_mat(self) -> DensityMatrix:
        # Init place holders:
        i = MatOfMats.Dims()
        j = MatOfMats.Dims()
        # Gat dimension of small inner matrix:
        d = self.dims.Sub
        # Type of inner variables:
        example = self[0,0]
        dtype = example.dtype
        # Init full matrix:
        mat = DensityMatrix.zeros(mat_dim=self.dims.Total, dtype=dtype)
        for i.Super, j.Super in self.all_indices():
            sub_mat = self[i.Super, j.Super]
            for i.Sub, j.Sub in sub_mat.all_indices():
                i.Total = i.Super*d + i.Sub
                j.Total = j.Super*d + j.Sub
                # Assign value:
                mat[i.Total, j.Total] = sub_mat[i.Sub, j.Sub]
        # Return:
        return mat

    def transpose(self) -> MatOfMats :
        new = np.transpose(self)
        new._copy_attributes(self)
        return new


def partial_transpose(
    mat_in:DensityMatrix, 
    num_qubits_on_first_part:typ.Optional[int]=None, 
    part_to_transpose:typ.Literal['first', 'second']='first'
) -> DensityMatrix:
    """partial_transpose Computes the partial transpose of a matrix.

    This is done in order to compute the negativity of a density-matrix.    

    Args:
        mat (DensityMatrix)
        num_qubits_on_first_part (typ.Optional[int], optional): The amount of qubits in the partition, belonging to the first part. The second part will get the rest. 
            Defaults to half the amount of qubits in the system.
        part_to_transpose (typ.Literal[&#39;first&#39;, &#39;second&#39;], optional): _description_. Defaults to 'first'.

    Returns:
        DensityMatrix
    """
    # Check input and assign default values:
    if num_qubits_on_first_part is None:
        num_qubits_on_first_part = _half_the_qubits(mat_in)
    assert num_qubits_on_first_part<mat_in.num_qubits

    ## Create matrix of matrices:
    super_mat_in = MatOfMats.from_mat(mat_in, num_qubits_on_first_part)

    ## transpose:
    if part_to_transpose=='first': 
        super_mat_out = super_mat_in.transpose()
    elif part_to_transpose=='second': 
        super_mat_out = super_mat_in.copy()
        for i, j in super_mat_out.all_indices():
            super_mat_out[i,j] = np.transpose(super_mat_in[i,j])
    else:
        raise Errors.SwitchCaseError(f"No such option for part '{part_to_transpose}'")

    ## Back to full matrix:
    mat_out = super_mat_out.to_full_mat()
    return mat_out


class CommonDensityMatrices():

    @staticmethod
    def plus_zero() -> DensityMatrix :
        rho = DensityMatrix([
            [1, 0, 1, 0],
            [0, 0, 0, 0],
            [1, 0, 1, 0],
            [0, 0, 0, 0]
        ]) * (1/2)    
        return rho 
        
    @staticmethod
    def bell(number: int = 1) -> DensityMatrix:
        if number == 1:
            rho = DensityMatrix([
                [1, 0, 0, 1],
                [0, 0, 0, 0],
                [0, 0, 0, 0],
                [1, 0, 0, 1]
            ]) * 1/2
        elif (number == 2) or (number == 3) or (number == 4):
            raise NotImplementedError("Other bell states are not yet supported")
        else:
            raise ValueError("Only valid bell states are 1,2,3 and 4.")
        return rho

    @staticmethod
    def ghz( num_qubits:int=3) -> DensityMatrix:
        zeros = [0]*num_qubits
        ones  = [1]*num_qubits
        ket = ( Ket(*zeros) + Ket(*ones) )*(1/np.sqrt(2))
        rho = DensityMatrix.from_ket(ket)
        return rho


def __test_mat(index: int) -> DensityMatrix:
    if   index == 1:
        ket = (1/np.sqrt(2))*(Ket(0,0)+Ket(1,1))
        mat = DensityMatrix.from_ket(ket)
    elif index == 2:
        mat = CommonDensityMatrices.bell(1)
    elif index == 3:
        mat = CommonDensityMatrices.ghz(3)
    return mat
