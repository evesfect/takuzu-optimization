from qiskit_optimization import QuadraticProgram
from qiskit_optimization.converters import LinearEqualityToPenalty

class Takuzu1DBuilder:
    def __init__(self, N, penalty_weight=10):
        self.N = N
        self.P = penalty_weight
        self.qp = QuadraticProgram()
        self.aux_vars = []
        
    def build(self):
        # 1. Define Primary Variables for the row
        for i in range(self.N):
            self.qp.binary_var(f'x{i}')
            
        # 2. Apply Cardinality Constraint (Equal 0s and 1s)
        # sum(x_i) == N/2
        linear_eq = {f'x{i}': 1 for i in range(self.N)}
        self.qp.linear_constraint(linear=linear_eq, sense='==', rhs=self.N // 2, name='card')
        
        # Convert the linear constraint natively into a squared QUBO penalty
        converter = LinearEqualityToPenalty(penalty=self.P)
        self.qp = converter.convert(self.qp)
        
        # 3. Apply Max 3-SAT "No Three Consecutive" via Rosenberg's Method
        self._apply_rosenberg_quadratization()
        
        return self.qp
        
    def _apply_rosenberg_quadratization(self):
        # We need to penalize (1,1,1) and (0,0,0) windows
        for i in range(self.N - 2):
            x_i = f'x{i}'
            x_j = f'x{i+1}'
            x_k = f'x{i+2}'
            
            # CLAUSE 1: Penalize (1, 1, 1). HUBO term is (x_i * x_j * x_k)
            # ROSENBERG REDUCTION: Let w = x_i * x_j
            w_name = f'w_{i}_{i+1}'
            self.qp.binary_var(w_name)
            self.aux_vars.append(w_name)
            
            # Add the reduced quadratic term: (w * x_k) * Penalty
            # And add the Rosenberg penalty: P * (3w + x_i*x_j - 2x_i*w - 2x_j*w)
            self.qp.minimize(
                linear={w_name: 3 * self.P},
                quadratic={
                    (w_name, x_k): self.P,       # The substituted term
                    (x_i, x_j): self.P,          # + x_i*x_j
                    (x_i, w_name): -2 * self.P,  # - 2x_i*w
                    (x_j, w_name): -2 * self.P   # - 2x_j*w
                }
            )

            # CLAUSE 2: Penalize (0, 0, 0). 
            # HUBO term is (1 - x_i)(1 - x_j)(1 - x_k)
            # Expanding this yields a cubic term -x_i*x_j*x_k. 
            # We reuse the exact same auxiliary variable 'w' here to reduce it!
            self.qp.minimize(
                linear={x_i: -self.P, x_j: -self.P, x_k: -self.P},
                quadratic={
                    (x_i, x_j): self.P,
                    (x_i, x_k): self.P,
                    (x_j, x_k): self.P,
                    (w_name, x_k): -self.P # The substituted -x_i*x_j*x_k
                }
            )
            # Add a constant offset for the expanded '1'
            self.qp.objective.constant += self.P

if __name__ == "__main__":
    N = 4 # Test on a 4-cell row
    builder = Takuzu1DBuilder(N=N)
    qubo = builder.build()
    
    print(f"Generated QUBO for 1D Takuzu (N={N})")
    print(f"Primary Variables: {N}")
    print(f"Auxiliary Variables Introduced: {len(builder.aux_vars)}")
    print(f"Total QUBO Variables: {qubo.get_num_vars()}")