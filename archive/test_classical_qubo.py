import numpy as np
import neal
from takuzu_2d_qubo import Takuzu2DBuilder
from qiskit_optimization.converters import QuadraticProgramToQubo

def verify_qubo_with_annealer(N=4):
    print(f"Building QUBO for {N}x{N} Takuzu...")
    builder = Takuzu2DBuilder(N=N)
    qp = builder.build()
    
    # Qiskit might have mixed constraints, force it to a pure QUBO
    conv = QuadraticProgramToQubo()
    qubo_problem = conv.convert(qp)
    
    # Extract linear and quadratic dictionaries for D-Wave
    linear = qubo_problem.objective.linear.to_dict()
    quadratic = qubo_problem.objective.quadratic.to_dict()
    
    # Map Qiskit's integer variable indices back to string names
    var_names = [v.name for v in qubo_problem.variables]
    
    Q = {}
    for i, val in linear.items():
        Q[(var_names[i], var_names[i])] = val
        
    for (i, j), val in quadratic.items():
        Q[(var_names[i], var_names[j])] = val

    print("Running D-Wave Simulated Annealing...")
    sampler = neal.SimulatedAnnealingSampler()
    
    # Run 1000 sweeps to guarantee we find the true ground state
    sampleset = sampler.sample_qubo(Q, num_reads=1000)
    best_sample = sampleset.first.sample
    best_energy = sampleset.first.energy

    print(f"\nLowest Energy Found: {best_energy}")
    
    # Extract the primary grid variables
    grid = np.zeros((N, N), dtype=int)
    for r in range(N):
        for c in range(N):
            var_name = f'x_{r}_{c}'
            grid[r, c] = best_sample.get(var_name, 0)
            
    print("\nGenerated Takuzu Board:")
    print(grid)

if __name__ == "__main__":
    verify_qubo_with_annealer(N=4)