class TakuzuQuboGenerator:
    def __init__(self, N=4, penalty=10):
        if N % 2 != 0:
            raise ValueError("Takuzu grid dimension N must be an even number.")
        self.N = N
        self.P = penalty
        self.Q = {}
        self.offset = 0.0

    def _add_lin(self, v, val):
        k = (v, v)
        self.Q[k] = self.Q.get(k, 0.0) + val

    def _add_quad(self, v1, v2, val):
        if v1 == v2:
            self._add_lin(v1, val)
            return
        k = tuple(sorted([v1, v2]))
        self.Q[k] = self.Q.get(k, 0.0) + val

    def build(self):
        # 1. Apply Cardinality (Equal 0s and 1s)
        for r in range(self.N):
            self._apply_exact_sum([f'x_{r}_{c}' for c in range(self.N)])
        for c in range(self.N):
            self._apply_exact_sum([f'x_{r}_{c}' for r in range(self.N)])

        # 2. Apply Consecutive Limits (No 111 or 000)
        for r in range(self.N):
            for c in range(self.N - 2):
                self._add_triplet(f'x_{r}_{c}', f'x_{r}_{c+1}', f'x_{r}_{c+2}', f'r{r}_c{c}')
        for c in range(self.N):
            for r in range(self.N - 2):
                self._add_triplet(f'x_{r}_{c}', f'x_{r+1}_{c}', f'x_{r+2}_{c}', f'c{c}_r{r}')

        return self.Q, self.offset

    def _apply_exact_sum(self, variables):
        target = self.N // 2
        lin_coef = self.P * (1 - 2 * target)
        
        for v in variables:
            self._add_lin(v, lin_coef)
        for i in range(len(variables)):
            for j in range(i + 1, len(variables)):
                self._add_quad(variables[i], variables[j], 2 * self.P)
                
        self.offset += self.P * (target ** 2)

    def _add_triplet(self, x_i, x_j, x_k, prefix):
        w = f'w_{prefix}'
        
        self._add_lin(w, 3 * self.P)
        self._add_quad(x_i, x_j, self.P)
        self._add_quad(x_i, w, -2 * self.P)
        self._add_quad(x_j, w, -2 * self.P)
        
        self._add_quad(w, x_k, self.P)
        
        self._add_lin(x_i, -self.P)
        self._add_lin(x_j, -self.P)
        self._add_lin(x_k, -self.P)
        self._add_quad(x_i, x_j, self.P)
        self._add_quad(x_i, x_k, self.P)
        self._add_quad(x_j, x_k, self.P)
        self._add_quad(w, x_k, -self.P) 
        
        self.offset += self.P