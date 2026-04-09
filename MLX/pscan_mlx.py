import math
import mlx.core as mx

"""
Optimized parallel scan algorithm in MLX (Blelloch version).

This replaces the naive sequential way of computing the sequence H[t] = A[t] * H[t-1] + X[t]
by computing it in parallel across time steps in O(log T) steps rather than O(T).

By wrapping the index-based sweep operations with `@mx.compile`, the MLX compute graph
fuses the operations, delivering orders of magnitude faster execution compared to the
uncompiled version. It handles non-power-of-2 sequence lengths seamlessly via padding.
"""

@mx.compile
def _compiled_pscan_indices(A, X):
    """
    Compiled version of the index-based Blelloch parallel scan.
    Orders of magnitude faster than uncompiled slicing.
    Must be called with L as a power of 2.
    """
    B, D, L, N = A.shape
    num_steps = int(math.log2(L))

    # Up sweep
    for k in range(0, num_steps):
        temp = 2**(k+1)
        A_temp = A[:, :, temp-1::temp]
        X_temp = X[:, :, temp-1::temp]
        A_k = A[:, :, 2**k-1::temp]
        X_k = X[:, :, 2**k-1::temp]
        
        X[:, :, temp-1::temp] = X_temp + A_temp * X_k
        A[:, :, temp-1::temp] = A_temp * A_k

    # Down sweep
    for k in range(num_steps, -1, -1):
        temp = 2**(k+1)
        A_temp = A[:, :, 3*2**k-1::temp]
        X_temp = X[:, :, 3*2**k-1::temp]
        A_k = A[:, :, temp-1:L-2**k:temp]
        X_k = X[:, :, temp-1:L-2**k:temp]
        
        X[:, :, 3*2**k-1::temp] = X_temp + A_temp * X_k
        A[:, :, 3*2**k-1::temp] = A_temp * A_k
        
    return X


def pscan_f(A, X):
    """
    A : (B, D, L, N)
    X : (B, D, L, N)

    H[t] = A[t] * H[t-1] + X[t] with H[0] = 0
    Returns the new sequence array.
    """
    B, D, L, N = A.shape
    
    # Pad to nearest power of 2 if necessary
    target_L = 2 ** math.ceil(math.log2(L))
    
    if target_L != L:
        pad_len = target_L - L
        # Pad A with 1s so it acts as identity for multiplication
        A_pad = mx.ones((B, D, pad_len, N))
        A_padded = mx.concatenate([A, A_pad], axis=2)
        # Pad X with 0s so it acts as identity for addition
        X_pad = mx.zeros((B, D, pad_len, N))
        X_padded = mx.concatenate([X, X_pad], axis=2)
        
        # Execute fast compiled index scan
        out_X = _compiled_pscan_indices(A_padded, X_padded)
        
        # Slice back to original length
        return out_X[:, :, :L, :]
    else:
        # Execute fast compiled index scan directly
        return _compiled_pscan_indices(A, X)


# main function, used in the Mamba model
def pscan(A_in, X_in):
    """
    Applies the parallel scan operation, as defined above. Returns a new tensor.

    Args:
        A_in : (B, L, ED, N)
        X_in : (B, L, ED, N)

    Returns:
        H : (B, L, ED, N)
    """

    A = A_in[:].transpose(0, 2, 1, 3)
    X = X_in[:].transpose(0, 2, 1, 3)

    X_out = pscan_f(A, X)

    return X_out.transpose(0, 2, 1, 3)
