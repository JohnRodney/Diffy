from typing import Callable, TypeVar, Union, Tuple, overload
from typing_extensions import Literal
import numpy as np
from numpy.typing import NDArray

# function type variable
F = TypeVar('F', bound=Callable[..., object])

# type variable for arrays  
T = TypeVar('T', bound=np.generic)
ScalarType = TypeVar('ScalarType', bound=np.generic)

# Kernel launch configuration types
GridType = Union[int, Tuple[int, ...]]
BlockType = Union[int, Tuple[int, ...]]

# CUDA kernel object that supports launch syntax
class CUDAKernel:
    def __getitem__(self, config: Union[GridType, Tuple[GridType, BlockType]]) -> Callable[..., None]: ...
    def __call__(self, *args, **kwargs) -> None: ...

# CUDA device arrays
class DeviceNDArray:
    def __init__(self, shape: Tuple[int, ...], dtype: Union[np.dtype[T], type[T], str]) -> None: ...
    @property
    def shape(self) -> Tuple[int, ...]: ...
    @property
    def size(self) -> int: ...
    @property
    def dtype(self) -> np.dtype[T]: ...
    def copy_to_host(self) -> NDArray[T]: ...
    # More specific indexing
    @overload
    def __getitem__(self, key: int) -> T: ...
    @overload
    def __getitem__(self, key: slice) -> DeviceNDArray: ...
    @overload
    def __getitem__(self, key: Tuple[Union[int, slice], ...]) -> Union[T, DeviceNDArray]: ...
    def __setitem__(self, key: Union[int, slice, Tuple[Union[int, slice], ...]], value: Union[T, DeviceNDArray]) -> None: ...

# CUDA JIT decorator - FIXED: distinguish device functions from kernels
@overload
def jit(signature_or_function: F, device: bool = False, inline: bool = ...) -> CUDAKernel: ...
@overload
def jit(signature_or_function: F, device: bool = True, inline: bool = ...) -> F: ...  # Device function returns original function
@overload
def jit(signature_or_function: str = ..., device: bool = False, inline: bool = ...) -> Callable[[F], CUDAKernel]: ...
@overload
def jit(signature_or_function: str = ..., device: bool = True, inline: bool = ...) -> Callable[[F], F]: ...

# CUDA Vector Types (from official docs)
class float32x2:
    def __init__(self, x: float, y: float) -> None: ...
    x: np.float32
    y: np.float32

class float32x3:
    def __init__(self, x: float, y: float, z: float) -> None: ...
    x: np.float32
    y: np.float32
    z: np.float32

class float32x4:
    def __init__(self, x: float, y: float, z: float, w: float) -> None: ...
    x: np.float32
    y: np.float32
    z: np.float32
    w: np.float32

class int32x2:
    def __init__(self, x: int, y: int) -> None: ...
    x: np.int32
    y: np.int32

class int32x3:
    def __init__(self, x: int, y: int, z: int) -> None: ...
    x: np.int32
    y: np.int32
    z: np.int32

class int64x2:
    def __init__(self, x: int, y: int) -> None: ...
    x: np.int64
    y: np.int64

class int64x3:
    def __init__(self, x: int, y: int, z: int) -> None: ...
    x: np.int64
    y: np.int64
    z: np.int64

# Convenience aliases (from docs)
float2 = float32x2
float3 = float32x3
float4 = float32x4
int2 = int32x2
int3 = int32x3
long2 = int64x2
long3 = int64x3

# CUDA grid and thread objects
class _BlockIdx:
    @property
    def x(self) -> int: ...
    @property
    def y(self) -> int: ...
    @property
    def z(self) -> int: ...

class _ThreadIdx:
    @property
    def x(self) -> int: ...
    @property
    def y(self) -> int: ...
    @property
    def z(self) -> int: ...

class _BlockDim:
    @property
    def x(self) -> int: ...
    @property
    def y(self) -> int: ...
    @property
    def z(self) -> int: ...

class _GridDim:
    @property
    def x(self) -> int: ...
    @property
    def y(self) -> int: ...
    @property
    def z(self) -> int: ...

blockIdx: _BlockIdx
threadIdx: _ThreadIdx
blockDim: _BlockDim
gridDim: _GridDim

# Missing kernel-only attributes from docs
laneid: int  # Thread index in current warp (0 to warpsize-1)
warpsize: int  # Size of warp (always 32)

@overload
def grid(ndim: Literal[1]) -> int: ...
@overload
def grid(ndim: Literal[2]) -> Tuple[int, int]: ...
@overload
def grid(ndim: Literal[3]) -> Tuple[int, int, int]: ...
@overload
def grid(ndim: int) -> Union[int, Tuple[int, ...]]: ...

def syncthreads() -> None: ...
def synchronize() -> None: ...  # Added: missing from our stubs

# Memory allocation - more specific dtype handling
@overload
def device_array(shape: Union[int, Tuple[int, ...]], dtype: type[np.float32] = ...) -> DeviceNDArray: ...
@overload
def device_array(shape: Union[int, Tuple[int, ...]], dtype: type[np.float64] = ...) -> DeviceNDArray: ...
@overload
def device_array(shape: Union[int, Tuple[int, ...]], dtype: type[np.int32] = ...) -> DeviceNDArray: ...
@overload
def device_array(shape: Union[int, Tuple[int, ...]], dtype: type[np.int64] = ...) -> DeviceNDArray: ...
@overload
def device_array(shape: Union[int, Tuple[int, ...]], dtype: Union[np.dtype[T], type[T], str] = ...) -> DeviceNDArray: ...

def device_array_like(array: Union[DeviceNDArray, NDArray[T]]) -> DeviceNDArray: ...
def to_device(array: NDArray[T]) -> DeviceNDArray: ...

# Memory management (from docs)
def pinned_memory(shape: Union[int, Tuple[int, ...]], dtype: Union[np.dtype[T], type[T], str] = ...) -> NDArray[T]: ...
def mapped_memory(shape: Union[int, Tuple[int, ...]], dtype: Union[np.dtype[T], type[T], str] = ...) -> NDArray[T]: ...
def managed_memory(shape: Union[int, Tuple[int, ...]], dtype: Union[np.dtype[T], type[T], str] = ...) -> DeviceNDArray: ...

# Streams (from docs)
class Stream:
    def synchronize(self) -> None: ...

def stream() -> Stream: ...

# Profiling (from docs)
def profile_stop() -> None: ...

# Device management
def select_device(device_id: int) -> None: ...

class Context:
    def synchronize(self) -> None: ...

def current_context() -> Context: ...

# Shared memory
class SharedArray:
    @overload
    def __getitem__(self, key: int) -> Union[np.float32, np.float64, np.int32, np.int64]: ...
    @overload
    def __getitem__(self, key: slice) -> SharedArray: ...
    @overload
    def __getitem__(self, key: Tuple[Union[int, slice], ...]) -> Union[np.float32, np.float64, np.int32, np.int64, SharedArray]: ...
    def __setitem__(self, key: Union[int, slice, Tuple[Union[int, slice], ...]], value: Union[np.float32, np.float64, np.int32, np.int64, SharedArray]) -> None: ...

class shared:
    @staticmethod
    def array(shape: Union[int, Tuple[int, ...]], dtype: Union[type[np.float32], type[np.float64], type[np.int32], type[np.int64], str], ndim: int = 1) -> SharedArray: ...

# Events and synchronization
class Event:
    def record(self) -> None: ...
    def synchronize(self) -> None: ...

def event() -> Event: ...
def event_elapsed_time(start: Event, end: Event) -> float: ...

# Atomic operations - more specific
class atomic:
    @overload
    @staticmethod
    def add(array: DeviceNDArray, index: int, value: np.float32) -> np.float32: ...
    @overload
    @staticmethod
    def add(array: DeviceNDArray, index: int, value: np.float64) -> np.float64: ...
    @overload
    @staticmethod
    def add(array: DeviceNDArray, index: int, value: np.int32) -> np.int32: ...
    @overload
    @staticmethod
    def add(array: DeviceNDArray, index: int, value: np.int64) -> np.int64: ...

# Libdevice math functions (from official docs)
class libdevice:
    @staticmethod
    def fast_sinf(x: float) -> float: ...
    @staticmethod
    def fast_cosf(x: float) -> float: ...
    @staticmethod
    def fast_powf(x: float, y: float) -> float: ...
    @staticmethod
    def fast_expf(x: float) -> float: ...
    @staticmethod
    def fast_logf(x: float) -> float: ... 