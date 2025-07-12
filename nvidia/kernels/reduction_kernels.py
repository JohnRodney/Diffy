import math
from numba import cuda
import numba

# CUDA Warp and Thread Block Constants
WARP_SIZE = 32
HALF_WARP_SIZE = 16
QUARTER_WARP_SIZE = 8
EIGHTH_WARP_SIZE = 4
SIXTEENTH_WARP_SIZE = 2
SMALLEST_WARP_SIZE = 1

# Thread Block Sizes
BLOCK_SIZE_256 = 256
BLOCK_SIZE_128 = 128
BLOCK_SIZE_64 = 64

# Grid Configuration
THREADS_PER_BLOCK_256 = 512
THREADS_PER_BLOCK_128 = 256
THREADS_PER_BLOCK_64 = 128

GRID_SIZE_MULTIPLIER_256 = 512
GRID_SIZE_MULTIPLIER_128 = 256
GRID_SIZE_MULTIPLIER_64 = 128

# Reduction Limits
MAX_BLOCKS_LIMIT = 64
REDUCTION_DIVISOR = 2

@cuda.jit(device=True)
def warp_reduce_256(sdata, tid):
    sdata[tid] += sdata[tid + WARP_SIZE]
    sdata[tid] += sdata[tid + HALF_WARP_SIZE] 
    sdata[tid] += sdata[tid + QUARTER_WARP_SIZE]
    sdata[tid] += sdata[tid + EIGHTH_WARP_SIZE]
    sdata[tid] += sdata[tid + SIXTEENTH_WARP_SIZE]
    sdata[tid] += sdata[tid + SMALLEST_WARP_SIZE]

@cuda.jit(device=True)
def warp_reduce_128(sdata, tid):
    sdata[tid] += sdata[tid + HALF_WARP_SIZE]
    sdata[tid] += sdata[tid + QUARTER_WARP_SIZE]
    sdata[tid] += sdata[tid + EIGHTH_WARP_SIZE]
    sdata[tid] += sdata[tid + SIXTEENTH_WARP_SIZE]
    sdata[tid] += sdata[tid + SMALLEST_WARP_SIZE]

@cuda.jit(device=True)
def warp_reduce_64(sdata, tid):
    sdata[tid] += sdata[tid + QUARTER_WARP_SIZE]
    sdata[tid] += sdata[tid + EIGHTH_WARP_SIZE]
    sdata[tid] += sdata[tid + SIXTEENTH_WARP_SIZE]
    sdata[tid] += sdata[tid + SMALLEST_WARP_SIZE]

@cuda.jit
def gpu_reduce_sum_256(g_idata, g_odata, n):
    sdata = cuda.shared.array(BLOCK_SIZE_256, dtype=numba.float32)
    tid = cuda.threadIdx.x
    i = cuda.blockIdx.x * THREADS_PER_BLOCK_256 + tid
    gridSize = GRID_SIZE_MULTIPLIER_256 * cuda.gridDim.x
    
    sdata[tid] = numba.float32(0.0)
    while i < n:
        sdata[tid] += g_idata[i] + g_idata[i + BLOCK_SIZE_256]
        i += gridSize
    
    cuda.syncthreads()
    
    if tid < BLOCK_SIZE_128:
        sdata[tid] += sdata[tid + BLOCK_SIZE_128]
    cuda.syncthreads()
    
    if tid < BLOCK_SIZE_64:
        sdata[tid] += sdata[tid + BLOCK_SIZE_64]
    cuda.syncthreads()
    
    if tid < WARP_SIZE:
        warp_reduce_256(sdata, tid)
    
    if tid == 0:
        g_odata[cuda.blockIdx.x] = sdata[0]

@cuda.jit
def gpu_reduce_sum_128(g_idata, g_odata, n):
    sdata = cuda.shared.array(BLOCK_SIZE_128, dtype=numba.float32)
    tid = cuda.threadIdx.x
    i = cuda.blockIdx.x * THREADS_PER_BLOCK_128 + tid
    gridSize = GRID_SIZE_MULTIPLIER_128 * cuda.gridDim.x
    
    sdata[tid] = numba.float32(0.0)
    while i < n:
        sdata[tid] += g_idata[i] + g_idata[i + BLOCK_SIZE_128]
        i += gridSize
    
    cuda.syncthreads()
    
    if tid < BLOCK_SIZE_64:
        sdata[tid] += sdata[tid + BLOCK_SIZE_64]
    cuda.syncthreads()
    
    if tid < WARP_SIZE:
        warp_reduce_128(sdata, tid)
    
    if tid == 0:
        g_odata[cuda.blockIdx.x] = sdata[0]

@cuda.jit
def gpu_reduce_sum_64(g_idata, g_odata, n):
    sdata = cuda.shared.array(BLOCK_SIZE_64, dtype=numba.float32)
    tid = cuda.threadIdx.x
    i = cuda.blockIdx.x * THREADS_PER_BLOCK_64 + tid
    gridSize = GRID_SIZE_MULTIPLIER_64 * cuda.gridDim.x
    
    sdata[tid] = numba.float32(0.0)
    while i < n:
        sdata[tid] += g_idata[i] + g_idata[i + BLOCK_SIZE_64]
        i += gridSize
    
    cuda.syncthreads()
    
    if tid < WARP_SIZE:
        warp_reduce_64(sdata, tid)
    
    if tid == 0:
        g_odata[cuda.blockIdx.x] = sdata[0]

def launch_reduce_sum(input_array, output_array, block_size=BLOCK_SIZE_256):
    n = input_array.size
    max_blocks = min(MAX_BLOCKS_LIMIT, math.ceil(n / (block_size * REDUCTION_DIVISOR)))
    
    if block_size == BLOCK_SIZE_256:
        gpu_reduce_sum_256[max_blocks, block_size](input_array, output_array, n)
    elif block_size == BLOCK_SIZE_128:
        gpu_reduce_sum_128[max_blocks, block_size](input_array, output_array, n)
    elif block_size == BLOCK_SIZE_64:
        gpu_reduce_sum_64[max_blocks, block_size](input_array, output_array, n)
    
    cuda.synchronize()

def reduce_sum_gpu(input_array_gpu):
    n = input_array_gpu.size
    max_blocks = min(MAX_BLOCKS_LIMIT, math.ceil(n / THREADS_PER_BLOCK_256))
    
    if max_blocks == 1:
        result = cuda.device_array(1, dtype=input_array_gpu.dtype)
        launch_reduce_sum(input_array_gpu, result)
        return result[0]
    
    temp = cuda.device_array(max_blocks, dtype=input_array_gpu.dtype)
    launch_reduce_sum(input_array_gpu, temp)
    
    while temp.size > 1:
        next_blocks = min(MAX_BLOCKS_LIMIT, math.ceil(temp.size / THREADS_PER_BLOCK_256))
        if next_blocks == 1:
            result = cuda.device_array(1, dtype=temp.dtype)
            launch_reduce_sum(temp, result)
            return result[0]
        
        next_temp = cuda.device_array(next_blocks, dtype=temp.dtype)
        launch_reduce_sum(temp, next_temp)
        temp = next_temp
    
    return temp[0]

def compute_bias_gradients_gpu(gradient_matrix, bias_gradients):
    batch_size, features = gradient_matrix.shape
    
    for i in range(features):
        column = gradient_matrix[:, i]
        bias_gradients[i] = reduce_sum_gpu(column) 