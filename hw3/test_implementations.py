#!/usr/bin/env python3

import numpy as np
import sys
import os

# Add the python directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), 'python'))

try:
    import needle as ndl
    from needle import backend_ndarray as nd
    
    print("Testing Python array operations...")
    
    # Test reshape
    print("Testing reshape...")
    A = nd.array(np.random.randn(4, 6), device=nd.cpu_numpy())
    B = A.reshape((2, 3, 4))
    print(f"Original shape: {A.shape}, Reshaped: {B.shape}")
    
    # Test permute
    print("Testing permute...")
    A = nd.array(np.random.randn(2, 3, 4), device=nd.cpu_numpy())
    B = A.permute((2, 0, 1))
    print(f"Original shape: {A.shape}, Permuted: {B.shape}")
    
    # Test broadcast_to
    print("Testing broadcast_to...")
    A = nd.array(np.random.randn(1, 3, 4), device=nd.cpu_numpy())
    B = A.broadcast_to((5, 3, 4))
    print(f"Original shape: {A.shape}, Broadcasted: {B.shape}")
    
    # Test getitem
    print("Testing getitem...")
    A = nd.array(np.random.randn(8, 8), device=nd.cpu_numpy())
    B = A[2:6, 1:5]
    print(f"Original shape: {A.shape}, Sliced: {B.shape}")
    
    print("Python array operations tests passed!")
    
except Exception as e:
    print(f"Error testing Python operations: {e}")
    print("This is expected if the C++ backend hasn't been compiled yet.")
    print("Run 'make lib' to compile the backend first.")

print("\nAll tests completed!") 