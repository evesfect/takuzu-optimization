import time
import numpy as np
from qiskit_aer.primitives import Sampler
from qiskit_algorithms import QAOA
from qiskit_algorithms.optimizers import COBYLA
from qiskit_optimization import QuadraticProgram
from qiskit_optimization.algorithms import MinimumEigenOptimizer
from takuzu_model import TakuzuQuboGenerator

def run_qaoa_benchmark(N=4, p_depth=1, enforce_uniqueness=False, max_iter=10):
    print(f"--- QAOA Benchmark (N={N}, Uniqueness={enforce_uniqueness}, Depth p={p_depth}) ---")
    
    # 1. Generate the pure mathematical QUBO
    builder = TakuzuQuboGenerator(N=N, penalty=10, enforce_uniqueness=enforce_uniqueness)
    Q, offset = builder.build()
    
    # Extract unique variables
    vars_set = set()
    linear = {}
    quadratic = {}
    
    for (k1, k2), val in Q.items():
        vars_set.add(k1)
        vars_set.add(k2)
        if k1 == k2:
            linear[k1] = val
        else:
            quadratic[(k1, k2)] = val

    # 2. Safely load our exact math into Qiskit
    qp = QuadraticProgram()
    for v in sorted(list(vars_set)):
        qp.binary_var(v)
        
    qp.minimize(constant=offset, linear=linear, quadratic=quadratic)
    print(f"Matrix loaded. Total Qubits Required: {qp.get_num_vars()}")
    
    # 3. Setup QAOA
    # Note: COBYLA is sequential, so this will only use 1 core. 
    sampler = Sampler(
        backend_options={"method": "matrix_product_state"}, 
        run_options={"shots": 256}
    )
    
    print(f"Initializing COBYLA Optimizer (Max Iterations: {max_iter})...")
    optimizer = COBYLA(maxiter=max_iter)
    
    qaoa = QAOA(sampler=sampler, optimizer=optimizer, reps=p_depth)
    min_eigen_optimizer = MinimumEigenOptimizer(qaoa)
    
    # 4. Execute the Quantum Simulation
    print(f"Executing QAOA circuit...")
    start_time = time.time()
    result = min_eigen_optimizer.solve(qp)
    elapsed = time.time() - start_time
    
    print(f"\nExecution Time: {elapsed:.2f} seconds")
    print(f"Lowest Energy Found (Cost): {result.fval}")
    
    # 5. Extract and print the board
    grid = np.zeros((N, N), dtype=int)
    for r in range(N):
        for c in range(N):
            var_name = f'x_{r}_{c}'
            if var_name in result.variables_dict:
                grid[r, c] = int(result.variables_dict[var_name])
                
    print("Generated Takuzu Board:")
    print(grid)
    print("-" * 60 + "\n")

if __name__ == "__main__":
    # Test 1: Shallow quantum circuit (p=1)
    run_qaoa_benchmark(N=4, p_depth=1, enforce_uniqueness=False, max_iter=10)
    
    # Test 2: Deeper quantum circuit (p=3)
    run_qaoa_benchmark(N=4, p_depth=3, enforce_uniqueness=False, max_iter=10)