import numpy as np
import neal
from takuzu_model import TakuzuQuboGenerator

def run_classical_baseline(N=4):
    print(f"Initializing Takuzu Model (N={N})...")
    builder = TakuzuQuboGenerator(N=N)
    Q, offset = builder.build()
    
    print(f"Matrix built. Total Variables (Primary + Auxiliary): {len(set([k[0] for k in Q.keys()]))}")
    print("Running D-Wave Simulated Annealing Baseline...")
    
    sampler = neal.SimulatedAnnealingSampler()
    sampleset = sampler.sample_qubo(Q, num_reads=5000)
    best_sample = sampleset.first.sample
    
    true_energy = sampleset.first.energy + offset
    print(f"\nLowest Energy Found (Cost): {true_energy}")
    
    grid = np.zeros((N, N), dtype=int)
    for r in range(N):
        for c in range(N):
            grid[r, c] = best_sample.get(f'x_{r}_{c}', 0)
            
    print("\nGenerated Takuzu Board:")
    print(grid)

if __name__ == "__main__":
    run_classical_baseline(N=4)