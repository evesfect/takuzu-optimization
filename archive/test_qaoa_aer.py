from qiskit_aer.primitives import Sampler
from qiskit_algorithms import QAOA
from qiskit_algorithms.optimizers import COBYLA
from qiskit_optimization import QuadraticProgram
from qiskit_optimization.algorithms import MinimumEigenOptimizer

def test_qaoa_pipeline():
    # 1. Create a dummy Quadratic Program (QUBO)
    qp = QuadraticProgram()
    qp.binary_var('x0')
    qp.binary_var('x1')
    
    # Simple objective: minimize x0 + x1 - 2*x0*x1 (Minimum is x0=1, x1=1)
    qp.minimize(linear={'x0': 1, 'x1': 1}, quadratic={('x0', 'x1'): -2})
    
    # 2. Setup the Sampler primitive using Aer
    # We specify shots here to simulate classical hardware behavior
    sampler = Sampler(run_options={"shots": 1024})
    
    # 3. Setup QAOA with COBYLA optimizer and circuit depth p=1
    optimizer = COBYLA(maxiter=50)
    qaoa = QAOA(sampler=sampler, optimizer=optimizer, reps=1)
    
    # 4. Wrap in a MinimumEigenOptimizer to map QUBO to Ising automatically
    min_eigen_optimizer = MinimumEigenOptimizer(qaoa)
    result = min_eigen_optimizer.solve(qp)
    
    print(f"QAOA Optimal Result: {result.x}")
    print(f"Objective Value: {result.fval}")

if __name__ == "__main__":
    print("Testing Qiskit QAOA Pipeline...")
    test_qaoa_pipeline()