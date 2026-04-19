import numpy as np
from qiskit_aer.primitives import Sampler
from qiskit_algorithms import QAOA
from qiskit_algorithms.optimizers import COBYLA
from qiskit_optimization.algorithms import MinimumEigenOptimizer
from takuzu_2d_qubo import Takuzu2DBuilder

def solve_qaoa_2d_grid(N=4):
    print(f"Building QUBO for {N}x{N} Takuzu...")
    builder = Takuzu2DBuilder(N=N)
    qp = builder.build()
    
    print(f"Total QUBO Variables: {qp.get_num_vars()}")
    print("Solving via QAOA (COBYLA Optimizer) to avoid memory explosion...")
    
    # 1. Setup the Sampler primitive using Aer with MPS compression directly
    sampler = Sampler(
        backend_options={"method": "matrix_product_state"}, 
        run_options={"shots": 1024}
    )
    
    # 2. Setup QAOA. We use a low maxiter to keep the run time reasonable for testing.
    optimizer = COBYLA(maxiter=100)
    qaoa = QAOA(sampler=sampler, optimizer=optimizer, reps=1)
    
    # 3. Solve the QUBO
    min_eigen_optimizer = MinimumEigenOptimizer(qaoa)
    result = min_eigen_optimizer.solve(qp)
    
    print(f"\nObjective Value (Cost): {result.fval}")
    if result.fval > 0.0:
        print("WARNING: Ground state cost is > 0. QAOA found an approximate (invalid) state.")
    else:
        print("SUCCESS: Valid Takuzu board found (Cost = 0.0).")
        
    # Extract the primary grid variables and format them into a 2D array
    grid = np.zeros((N, N), dtype=int)
    for r in range(N):
        for c in range(N):
            var_name = f'x_{r}_{c}'
            # QAOA returns floats (e.g., 1.0 or 0.0), so we cast to int
            grid[r, c] = int(result.variables_dict[var_name])
            
    print("\nGenerated Takuzu Board:")
    print(grid)

if __name__ == "__main__":
    solve_qaoa_2d_grid(N=4)