from qiskit_optimization import QuadraticProgram
from qiskit_optimization.converters import LinearEqualityToPenalty

class Takuzu2DBuilder:
    def __init__(self, N, penalty_weight=10):
        if N % 2 != 0:
            raise ValueError("Takuzu grid dimension N must be an even number.")
        self.N = N
        self.P = penalty_weight
        self.qp = QuadraticProgram()
        self.aux_vars = []
        
        # Accumulators to prevent Qiskit from overwriting the objective
        self.lin_dict = {}
        self.quad_dict = {}
        self.const = 0.0

    def _add_lin(self, var, val):
        self.lin_dict[var] = self.lin_dict.get(var, 0.0) + val
        
    def _add_quad(self, v1, v2, val):
        # Sort keys to prevent duplicate edge generation
        k = tuple(sorted([v1, v2]))
        self.quad_dict[k] = self.quad_dict.get(k, 0.0) + val

    def build(self):
        # 1. Define Primary Variables
        for r in range(self.N):
            for c in range(self.N):
                self.qp.binary_var(f'x_{r}_{c}')
                
        # 2. Apply Cardinality Constraints
        self._apply_cardinality()
        
        # 3. Apply Max 3-SAT "No Three Consecutive" Limits
        self._apply_consecutive_limits()
        
        # 4. Set the complete objective EXACTLY ONCE
        self.qp.minimize(
            constant=self.const,
            linear=self.lin_dict,
            quadratic=self.quad_dict
        )
        
        return self.qp

    def _apply_cardinality(self):
        for r in range(self.N):
            linear_eq = {f'x_{r}_{c}': 1 for c in range(self.N)}
            self.qp.linear_constraint(linear=linear_eq, sense='==', rhs=self.N // 2, name=f'card_row_{r}')
            
        for c in range(self.N):
            linear_eq = {f'x_{r}_{c}': 1 for r in range(self.N)}
            self.qp.linear_constraint(linear=linear_eq, sense='==', rhs=self.N // 2, name=f'card_col_{c}')
            
        # Convert to penalties
        converter = LinearEqualityToPenalty(penalty=self.P)
        self.qp = converter.convert(self.qp)
        
        # Extract the converted objective so we can safely add Rosenberg terms to it
        lin_terms = self.qp.objective.linear.to_dict(use_name=True)
        quad_terms = self.qp.objective.quadratic.to_dict(use_name=True)
        
        for var, coef in lin_terms.items():
            self._add_lin(var, coef)
            
        for (v1, v2), coef in quad_terms.items():
            self._add_quad(v1, v2, coef)
            
        self.const += self.qp.objective.constant

    def _apply_consecutive_limits(self):
        for r in range(self.N):
            for c in range(self.N - 2):
                self._add_rosenberg_triplet(
                    f'x_{r}_{c}', f'x_{r}_{c+1}', f'x_{r}_{c+2}', 
                    prefix=f'row_{r}_col_{c}'
                )
                
        for c in range(self.N):
            for r in range(self.N - 2):
                self._add_rosenberg_triplet(
                    f'x_{r}_{c}', f'x_{r+1}_{c}', f'x_{r+2}_{c}', 
                    prefix=f'col_{c}_row_{r}'
                )

    def _add_rosenberg_triplet(self, x_i, x_j, x_k, prefix):
        w_name = f'w_{prefix}'
        self.qp.binary_var(w_name)
        self.aux_vars.append(w_name)
        
        # Clause 1 (Penalize 1,1,1) + Rosenberg Equivalence Penalty
        self._add_lin(w_name, 3 * self.P)
        self._add_quad(x_i, x_j, self.P)
        self._add_quad(x_i, w_name, -2 * self.P)
        self._add_quad(x_j, w_name, -2 * self.P)
        self._add_quad(w_name, x_k, self.P)

        # Clause 2 (Penalize 0,0,0)
        self._add_lin(x_i, -self.P)
        self._add_lin(x_j, -self.P)
        self._add_lin(x_k, -self.P)
        
        self._add_quad(x_i, x_j, self.P)
        self._add_quad(x_i, x_k, self.P)
        self._add_quad(x_j, x_k, self.P)
        self._add_quad(w_name, x_k, -self.P)
        
        self.const += self.P

if __name__ == "__main__":
    builder = Takuzu2DBuilder(N=4)
    qubo = builder.build()
    print("Builder updated and compiled.")