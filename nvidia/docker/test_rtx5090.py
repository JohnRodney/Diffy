#!/usr/bin/env python3
"""
RTX 5090 compatibility test script
Run this before launching the full parameter sweep
"""

import sys
import numpy as np

def test_basic_imports():
    """Test basic imports"""
    print("🔍 Testing basic imports...")
    try:
        import numba
        print(f"✅ Numba version: {numba.__version__}")
    except ImportError as e:
        print(f"❌ Numba import failed: {e}")
        return False
    
    try:
        import numpy as np
        print(f"✅ NumPy version: {np.__version__}")
    except ImportError as e:
        print(f"❌ NumPy import failed: {e}")
        return False
    
    return True

def test_cuda_availability():
    """Test CUDA availability"""
    print("\n🔍 Testing CUDA availability...")
    try:
        from numba import cuda
        
        if not cuda.is_available():
            print("❌ CUDA is not available")
            return False
        
        print("✅ CUDA is available")
        
        # List CUDA devices
        devices = cuda.list_devices()
        print(f"✅ Found {len(devices)} CUDA devices:")
        for i, device in enumerate(devices):
            print(f"   Device {i}: {device}")
        
        return True
        
    except Exception as e:
        print(f"❌ CUDA test failed: {e}")
        return False

def test_gpu_memory():
    """Test GPU memory allocation"""
    print("\n🔍 Testing GPU memory allocation...")
    try:
        from numba import cuda
        
        # Test small allocation
        test_size = 1024
        gpu_array = cuda.device_array(test_size, dtype=np.float32)
        print(f"✅ Successfully allocated {test_size} floats on GPU")
        
        # Test larger allocation (1MB)
        large_size = 1024 * 1024
        gpu_large = cuda.device_array(large_size, dtype=np.float32)
        print(f"✅ Successfully allocated {large_size} floats on GPU (~4MB)")
        
        # Test matrix allocation
        matrix_size = (1024, 1024)
        gpu_matrix = cuda.device_array(matrix_size, dtype=np.float32)
        print(f"✅ Successfully allocated {matrix_size} matrix on GPU (~4MB)")
        
        return True
        
    except Exception as e:
        print(f"❌ GPU memory test failed: {e}")
        return False

def test_simple_kernel():
    """Test a simple CUDA kernel"""
    print("\n🔍 Testing simple CUDA kernel...")
    try:
        from numba import cuda
        
        @cuda.jit
        def add_kernel(a, b, c):
            idx = cuda.grid(1)
            if idx < a.size:
                c[idx] = a[idx] + b[idx]
        
        # Create test data
        size = 1000
        a = np.random.random(size).astype(np.float32)
        b = np.random.random(size).astype(np.float32)
        c = np.zeros_like(a)
        
        # Transfer to GPU
        a_gpu = cuda.to_device(a)
        b_gpu = cuda.to_device(b)
        c_gpu = cuda.to_device(c)
        
        # Launch kernel
        threads_per_block = 256
        blocks_per_grid = (size + threads_per_block - 1) // threads_per_block
        add_kernel[blocks_per_grid, threads_per_block](a_gpu, b_gpu, c_gpu)
        
        # Get result
        result = c_gpu.copy_to_host()
        
        # Verify result
        expected = a + b
        if np.allclose(result, expected):
            print("✅ Simple kernel test passed")
            return True
        else:
            print("❌ Simple kernel test failed - wrong results")
            return False
            
    except Exception as e:
        print(f"❌ Simple kernel test failed: {e}")
        return False

def test_matrix_multiplication():
    """Test matrix multiplication kernel"""
    print("\n🔍 Testing matrix multiplication kernel...")
    try:
        from numba import cuda
        
        @cuda.jit
        def matmul_kernel(A, B, C):
            row, col = cuda.grid(2)
            if row < C.shape[0] and col < C.shape[1]:
                temp = 0.0
                for k in range(A.shape[1]):
                    temp += A[row, k] * B[k, col]
                C[row, col] = temp
        
        # Create test matrices
        size = 64  # Small test
        A = np.random.random((size, size)).astype(np.float32)
        B = np.random.random((size, size)).astype(np.float32)
        C = np.zeros((size, size), dtype=np.float32)
        
        # Transfer to GPU
        A_gpu = cuda.to_device(A)
        B_gpu = cuda.to_device(B)
        C_gpu = cuda.to_device(C)
        
        # Launch kernel
        threads_per_block = (16, 16)
        blocks_per_grid_x = (size + threads_per_block[0] - 1) // threads_per_block[0]
        blocks_per_grid_y = (size + threads_per_block[1] - 1) // threads_per_block[1]
        blocks_per_grid = (blocks_per_grid_x, blocks_per_grid_y)
        
        matmul_kernel[blocks_per_grid, threads_per_block](A_gpu, B_gpu, C_gpu)
        
        # Get result
        result = C_gpu.copy_to_host()
        
        # Verify with NumPy
        expected = np.dot(A, B)
        if np.allclose(result, expected, rtol=1e-5):
            print("✅ Matrix multiplication test passed")
            return True
        else:
            print("❌ Matrix multiplication test failed - wrong results")
            return False
            
    except Exception as e:
        print(f"❌ Matrix multiplication test failed: {e}")
        return False

def get_gpu_info():
    """Get detailed GPU information"""
    print("\n🔍 Getting GPU information...")
    try:
        from numba import cuda
        
        context = cuda.current_context()
        device = context.device
        
        print(f"✅ GPU Name: {device.name}")
        print(f"✅ Compute Capability: {device.compute_capability}")
        print(f"✅ Total Memory: {device.total_memory / (1024**3):.2f} GB")
        print(f"✅ Multiprocessor Count: {device.MULTIPROCESSOR_COUNT}")
        print(f"✅ Warp Size: {device.WARP_SIZE}")
        print(f"✅ Max Threads Per Block: {device.MAX_THREADS_PER_BLOCK}")
        
        return True
        
    except Exception as e:
        print(f"❌ GPU info failed: {e}")
        return False

def main():
    """Run all RTX 5090 compatibility tests"""
    print("🚀 RTX 5090 Compatibility Test Suite")
    print("=" * 40)
    
    tests = [
        ("Basic Imports", test_basic_imports),
        ("CUDA Availability", test_cuda_availability),
        ("GPU Memory", test_gpu_memory),
        ("Simple Kernel", test_simple_kernel),
        ("Matrix Multiplication", test_matrix_multiplication),
        ("GPU Information", get_gpu_info)
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"\n{'='*20} {test_name} {'='*20}")
        success = test_func()
        results.append((test_name, success))
    
    # Summary
    print("\n" + "="*60)
    print("🎯 TEST SUMMARY")
    print("="*60)
    
    passed = 0
    for test_name, success in results:
        status = "✅ PASSED" if success else "❌ FAILED"
        print(f"{test_name:25} {status}")
        if success:
            passed += 1
    
    print(f"\nTotal: {passed}/{len(tests)} tests passed")
    
    if passed == len(tests):
        print("\n🎉 All tests passed! RTX 5090 is ready for parameter sweep!")
        sys.exit(0)
    else:
        print(f"\n⚠️  {len(tests) - passed} tests failed. Check RTX 5090 setup.")
        sys.exit(1)

if __name__ == "__main__":
    main() 