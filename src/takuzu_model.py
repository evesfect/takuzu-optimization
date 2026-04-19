class TakuzuQuboGenerator:
    def __init__(self, N=4, penalty=10, enforce_uniqueness=True):
        if N % 2 != 0:
            raise ValueError("Takuzu grid dimension N must be an even number.")
        self.N = N
        self.P = penalty
        self.enforce_uniqueness = enforce_uniqueness
        self.Q = {}
        self.offset = 0.0
        self.aux_count = 0

    def _add_lin(self, v, val):
        k = (v, v)
        self.Q[k] = self.Q.get(k, 0.0) + val

    def _add_quad(self, v1, v2, val):
        if v1 == v2:
            self._add_lin(v1, val)
            return
        k = tuple(sorted([v1, v2]))
        self.Q[k] = self.Q.get(k, 0.0) + val

    def _get_aux(self, prefix):
        self.aux_count += 1
        return f"aux_{prefix}_{self.aux_count}"

    def build(self):
        for r in range(self.N):
            self._apply_exact_sum([f'x_{r}_{c}' for c in range(self.N)])
        for c in range(self.N):
            self._apply_exact_sum([f'x_{r}_{c}' for r in range(self.N)])

        for r in range(self.N):
            for c in range(self.N - 2):
                self._add_triplet(f'x_{r}_{c}', f'x_{r}_{c+1}', f'x_{r}_{c+2}', f'r{r}c{c}')
        for c in range(self.N):
            for r in range(self.N - 2):
                self._add_triplet(f'x_{r}_{c}', f'x_{r+1}_{c}', f'x_{r+2}_{c}', f'c{c}r{r}')

        if self.enforce_uniqueness:
            self._apply_uniqueness()

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
        w = self._get_aux(f"trip_{prefix}")
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

    def _gate_and(self, A, B, prefix):
        w = self._get_aux(f"and_{prefix}")
        self._add_quad(A, B, self.P)
        self._add_quad(A, w, -2 * self.P)
        self._add_quad(B, w, -2 * self.P)
        self._add_lin(w, 3 * self.P)
        return w

    def _gate_nor(self, A, B, prefix):
        w = self._get_aux(f"nor_{prefix}")
        self._add_quad(A, B, self.P)
        self._add_lin(A, -self.P)
        self._add_lin(B, -self.P)
        self._add_quad(A, w, 2 * self.P)
        self._add_quad(B, w, 2 * self.P)
        self._add_lin(w, -self.P)
        self.offset += self.P
        return w

    def _gate_or(self, A, B, prefix):
        w = self._get_aux(f"or_{prefix}")
        self._add_quad(A, B, self.P)
        self._add_quad(A, w, -2 * self.P)
        self._add_quad(B, w, -2 * self.P)
        self._add_lin(A, self.P)
        self._add_lin(B, self.P)
        self._add_lin(w, self.P)
        return w

    def _cascade_and(self, var_list, prefix):
        if len(var_list) == 1:
            return var_list[0]
        if len(var_list) == 2:
            return self._gate_and(var_list[0], var_list[1], prefix)
        mid = len(var_list) // 2
        left = self._cascade_and(var_list[:mid], f"{prefix}_L")
        right = self._cascade_and(var_list[mid:], f"{prefix}_R")
        return self._gate_and(left, right, f"{prefix}_M")

    def _apply_uniqueness(self):
        for r1 in range(self.N):
            for r2 in range(r1 + 1, self.N):
                match_vars = []
                for c in range(self.N):
                    A_and_B = self._gate_and(f'x_{r1}_{c}', f'x_{r2}_{c}', f"r{r1}r{r2}c{c}_11")
                    notA_and_notB = self._gate_nor(f'x_{r1}_{c}', f'x_{r2}_{c}', f"r{r1}r{r2}c{c}_00")
                    xnor = self._gate_or(A_and_B, notA_and_notB, f"r{r1}r{r2}c{c}_xnor")
                    match_vars.append(xnor)
                identical_row_flag = self._cascade_and(match_vars, f"row_match_{r1}_{r2}")
                self._add_lin(identical_row_flag, self.P)

        for c1 in range(self.N):
            for c2 in range(c1 + 1, self.N):
                match_vars = []
                for r in range(self.N):
                    A_and_B = self._gate_and(f'x_{r}_{c1}', f'x_{r}_{c2}', f"c{c1}c{c2}r{r}_11")
                    notA_and_notB = self._gate_nor(f'x_{r}_{c1}', f'x_{r}_{c2}', f"c{c1}c{c2}r{r}_00")
                    xnor = self._gate_or(A_and_B, notA_and_notB, f"c{c1}c{c2}r{r}_xnor")
                    match_vars.append(xnor)
                identical_col_flag = self._cascade_and(match_vars, f"col_match_{c1}_{c2}")
                self._add_lin(identical_col_flag, self.P)