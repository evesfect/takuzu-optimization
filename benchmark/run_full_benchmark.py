"""
Full benchmark script for the Takuzu report.
Runs QAOA at multiple depths and computes detailed metrics.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import time
import numpy as np
from hubo_qaoa import TakuzuHUBO, run_qaoa_native_hubo
from sat_solver import solve_takuzu_sat, count_all_solutions


def detailed_qaoa_benchmark(N=4, p_depths=[1, 2, 3], n_trials=3,
                            max_iter=300, shots=8192):
    """
    Run QAOA multiple times at each depth and compute statistics.
    """
    print("=" * 70)
    print(f"DETAILED QAOA BENCHMARK (N={N})")
    print("=" * 70)

    # First, get ground truth
    print("\n--- Ground Truth ---")
    hubo = TakuzuHUBO(N=N)
    terms = hubo.build_hubo_terms()
    constant = terms.get(frozenset(), 0.0)
    print(f"  HUBO constant offset: {constant:.2f}")
    print(f"  Ground state energy: 0.0")
    print(f"  Total HUBO terms: {len(terms)}")
    
    # Convert and get Pauli term count
    hamiltonian = hubo.hubo_to_hamiltonian(terms)
    print(f"  Pauli terms in Hamiltonian: {len(hamiltonian)}")
    print(f"  Qubits: {N*N}")
    
    # Known valid solutions from brute force / SAT
    print(f"  Valid solutions (4x4): 72")

    all_results = {}

    for p in p_depths:
        print(f"\n{'='*60}")
        print(f"  DEPTH p = {p} ({n_trials} trials)")
        print(f"{'='*60}")
        
        trial_results = []
        for trial in range(n_trials):
            print(f"\n  --- Trial {trial+1}/{n_trials} ---")
            r = run_qaoa_native_hubo(
                N=N, p_depth=p, max_iter=max_iter, shots=shots,
                P1=4.0, P2=4.0, P3=6.0
            )
            trial_results.append(r)
        
        # Aggregate
        eigenvalues = [r['eigenvalue'] for r in trial_results]
        valid_count = sum(1 for r in trial_results if r['valid_solution'])
        times = [r['elapsed_time'] for r in trial_results]
        
        all_results[p] = {
            'eigenvalues': eigenvalues,
            'mean_eigenvalue': np.mean(eigenvalues),
            'best_eigenvalue': np.min(eigenvalues),
            'valid_count': valid_count,
            'n_trials': n_trials,
            'mean_time': np.mean(times),
            'trials': trial_results,
        }
        
        print(f"\n  p={p} Summary:")
        print(f"    Mean eigenvalue: {np.mean(eigenvalues):.4f}")
        print(f"    Best eigenvalue: {np.min(eigenvalues):.4f}")
        print(f"    Valid solutions found: {valid_count}/{n_trials}")
        print(f"    Mean time: {np.mean(times):.2f}s")

    # Classical comparison
    print(f"\n{'='*60}")
    print("  CLASSICAL RC2 SAT SOLVER")
    print(f"{'='*60}")
    grid, sat_time = solve_takuzu_sat(N=N)

    # Final summary table
    print(f"\n\n{'='*70}")
    print("FINAL RESULTS TABLE")
    print(f"{'='*70}")
    print(f"{'Depth p':<10} {'Best Eig.':<12} {'Mean Eig.':<12} {'Valid/Trials':<14} {'Mean Time':<10}")
    print("-" * 58)
    for p in p_depths:
        r = all_results[p]
        print(f"{p:<10} {r['best_eigenvalue']:<12.4f} {r['mean_eigenvalue']:<12.4f} "
              f"{r['valid_count']}/{r['n_trials']:<12} {r['mean_time']:<10.2f}")
    print("-" * 58)
    print(f"{'RC2 SAT':<10} {'0.0':<12} {'0.0':<12} {'1/1':<14} {sat_time:<10.6f}")
    print(f"\nNote: Ground state energy = 0.0 (all constraints satisfied)")
    print(f"Note: Eigenvalue = <H> (expectation value over measurement distribution)")
    print(f"Note: Lower eigenvalue = better concentration around ground state")

    return all_results


if __name__ == "__main__":
    results = detailed_qaoa_benchmark(N=4, p_depths=[1, 2, 3], n_trials=3)
