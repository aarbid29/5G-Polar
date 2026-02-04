#!/usr/bin/env python3
"""
Generate parity check matrices for polar codes (N=32).
This creates the H matrices needed for the BiMamba decoder.
"""

import numpy as np
import os


def generate_polar_G_matrix(n):
    """
    Generate the generator matrix G for polar codes using Kronecker products.
    For N=32, this is F^⊗5 where F = [[1,0],[1,1]]
    """
    assert n & (n - 1) == 0, "n must be a power of 2"
    
    # Base matrix F
    F = np.array([[1, 0], [1, 1]], dtype=int)
    
    # Compute log2(n)
    m = int(np.log2(n))
    
    # Start with F
    G = F
    
    # Kronecker product m-1 more times
    for _ in range(m - 1):
        G = np.kron(G, F)
    
    return G


def get_frozen_set(n, k, method='polar_5g'):
    """
    Get the frozen bit positions for polar codes.
    Returns indices of the k most reliable positions.
    """
    if method == 'polar_5g':
        # 5G NR reliability sequence for different code lengths
        # This is a simplified version - real 5G uses precomputed sequences
        
        if n == 32:
            # Reliability sequence for N=32 (higher index = more reliable)
            # Based on Bhattacharyya parameters
            reliability_order = [
                0, 1, 2, 4, 8, 16, 3, 5, 9, 6, 17, 10, 18, 12, 20, 24,
                7, 11, 13, 19, 14, 21, 25, 22, 26, 28, 15, 23, 27, 29, 30, 31
            ]
        elif n == 64:
            reliability_order = list(range(64))  # Placeholder
            # In practice, use 5G NR sequence
        elif n == 128:
            reliability_order = list(range(128))  # Placeholder
        else:
            # Simple heuristic for other lengths
            reliability_order = list(range(n))
    else:
        # Simple bit-reversal ordering
        reliability_order = [int(bin(i)[2:].zfill(int(np.log2(n)))[::-1], 2) 
                            for i in range(n)]
    
    # Most reliable k positions for information bits
    info_positions = reliability_order[-k:]
    
    # Frozen positions are the rest
    frozen_positions = [i for i in range(n) if i not in info_positions]
    
    return sorted(info_positions), sorted(frozen_positions)


def generate_parity_check_matrix(n, k):
    """
    Generate parity check matrix H for polar codes.
    H = [G_frozen^T | I] in systematic form
    where G_frozen is the generator matrix rows corresponding to frozen bits
    """
    # Generate full generator matrix
    G = generate_polar_G_matrix(n)
    
    # Get frozen and info bit positions
    info_positions, frozen_positions = get_frozen_set(n, k)
    
    # Extract rows of G corresponding to frozen positions
    G_frozen = G[frozen_positions, :]  # Shape: (n-k, n)
    
    # Parity check matrix is the transpose of frozen part
    # This gives us (n-k) parity check equations
    H = G_frozen  # Shape: (n-k, n)
    
    return H, info_positions, frozen_positions


def save_parity_matrix(filename, H):
    """Save parity check matrix to text file"""
    np.savetxt(filename, H, fmt='%d', delimiter=' ')
    print(f"Saved: {filename}")
    print(f"  Shape: {H.shape} ({H.shape[0]} checks, {H.shape[1]} bits)")
    print(f"  Density: {np.sum(H) / H.size * 100:.1f}%")


def generate_all_polar_matrices():
    """Generate parity check matrices for common polar code configurations"""
    
    # Create codes directory
    os.makedirs('codes', exist_ok=True)
    
    # N=32 configurations
    configs_32 = [
        (32, 8),   # Rate 1/4
        (32, 16),  # Rate 1/2
        (32, 20),  # Rate 5/8
        (32, 24),  # Rate 3/4
    ]
    
    # N=64 configurations (optional)
    configs_64 = [
        (64, 16),  # Rate 1/4
        (64, 32),  # Rate 1/2
        (64, 48),  # Rate 3/4
    ]
    
    # N=128 configurations (optional)
    configs_128 = [
        (128, 64),  # Rate 1/2
    ]
    
    all_configs = configs_32 + configs_64 + configs_128
    
    print("Generating Polar Code Parity Check Matrices")
    print("=" * 60)
    
    for n, k in all_configs:
        print(f"\nGenerating POLAR N={n}, K={k} (Rate={k/n:.3f})...")
        
        # Generate H matrix
        H, info_pos, frozen_pos = generate_parity_check_matrix(n, k)
        
        # Save to file
        filename = f'codes/POLAR_N{n}_K{k}.txt'
        save_parity_matrix(filename, H)
        
        # Also save info about bit positions
        info_filename = f'codes/POLAR_N{n}_K{k}_positions.txt'
        with open(info_filename, 'w') as f:
            f.write(f"# Polar Code N={n}, K={k}\n")
            f.write(f"# Information bit positions:\n")
            f.write(' '.join(map(str, info_pos)) + '\n')
            f.write(f"# Frozen bit positions:\n")
            f.write(' '.join(map(str, frozen_pos)) + '\n')
        print(f"  Positions saved: {info_filename}")
    
    print("\n" + "=" * 60)
    print("Done! All matrices generated in 'codes/' directory")


def verify_matrix(n, k):
    """Verify that the parity check matrix is correct"""
    H, info_pos, frozen_pos = generate_parity_check_matrix(n, k)
    G = generate_polar_G_matrix(n)
    
    # For polar codes, H @ G^T should have specific structure
    # The frozen rows should satisfy certain properties
    
    print(f"\nVerifying POLAR N={n}, K={k}...")
    print(f"  H shape: {H.shape}")
    print(f"  G shape: {G.shape}")
    print(f"  Info positions ({len(info_pos)}): {info_pos[:10]}...")
    print(f"  Frozen positions ({len(frozen_pos)}): {frozen_pos[:10]}...")
    
    # Check that we have the right number of parity checks
    assert H.shape[0] == n - k, "Wrong number of parity check equations"
    assert H.shape[1] == n, "Wrong codeword length"
    
    # Verify using generator matrix
    G_info = G[info_pos, :]  # Information bit part
    
    # H @ c = 0 for all valid codewords c = u @ G_info (mod 2)
    # Test with a few random information vectors
    for _ in range(10):
        u = np.random.randint(0, 2, k)
        c = (u @ G_info) % 2
        syndrome = (H @ c) % 2
        if not np.all(syndrome == 0):
            print(f"  ❌ FAILED: Syndrome not zero!")
            return False
    
    print(f"  ✓ Verified: All test codewords satisfy H @ c = 0")
    return True


def show_example_matrix():
    """Show what the matrix looks like"""
    n, k = 32, 16
    H, info_pos, frozen_pos = generate_parity_check_matrix(n, k)
    
    print("\nExample: POLAR_N32_K16.txt content:")
    print("=" * 60)
    for i, row in enumerate(H[:5]):  # Show first 5 rows
        print(' '.join(map(str, row)))
    print("... (16 rows total)")
    print("=" * 60)
    
    print(f"\nThis is a {H.shape[0]}×{H.shape[1]} matrix")
    print(f"Each row is one parity check equation")
    print(f"Total: {H.shape[0]} parity checks for a (32,16) polar code")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate polar code parity check matrices")
    parser.add_argument('--verify', action='store_true', help="Verify generated matrices")
    parser.add_argument('--example', action='store_true', help="Show example matrix")
    parser.add_argument('--n', type=int, default=None, help="Code length (power of 2)")
    parser.add_argument('--k', type=int, default=None, help="Information length")
    
    args = parser.parse_args()
    
    if args.example:
        show_example_matrix()
    elif args.n and args.k:
        # Generate specific matrix
        os.makedirs('codes', exist_ok=True)
        print(f"Generating POLAR N={args.n}, K={args.k}...")
        H, info_pos, frozen_pos = generate_parity_check_matrix(args.n, args.k)
        filename = f'codes/POLAR_N{args.n}_K{args.k}.txt'
        save_parity_matrix(filename, H)
        if args.verify:
            verify_matrix(args.n, args.k)
    else:
        # Generate all standard matrices
        generate_all_polar_matrices()
        
        if args.verify:
            print("\nVerifying matrices...")
            verify_matrix(32, 16)
            verify_matrix(32, 24)
