"""
Native HUBO QAOA solver for Takuzu (N=4).

Constructs the cost Hamiltonian directly from the HUBO penalty terms
by mapping binary variables x_i -> (I - Z_i)/2. This avoids any
quadratization and operates on exactly N^2 = 16 qubits.
"""

import numpy as np
from itertools import combinations
from qiskit.quantum_info import SparsePauliOp
from qiskit_aer import AerSimulator
from qiskit.circuit.library import QAOAAnsatz
from qiskit.primitives import StatevectorEstimator
from qiskit_algorithms.minimum_eigensolvers import QAOA
from qiskit_algorithms.optimizers import COBYLA
from qiskit_aer.primitives import Sampler


class TakuzuHUBO:
    """Builds the HUBO cost function for an NxN Takuzu grid."""

    def __init__(self, N=4, P1=4.0, P2=4.0, P3=6.0):
        """
        Args:
            N: Grid size (must be even).
            P1: Penalty weight for consecutive constraint.
            P2: Penalty weight for cardinality constraint.
            P3: Penalty weight for uniqueness constraint.
        """
        if N % 2 != 0:
            raise ValueError("N must be even")
        self.N = N
        self.n_qubits = N * N
        self.P1 = P1
        self.P2 = P2
        self.P3 = P3

    def var_index(self, r, c):
        """Map grid position (r, c) to qubit index."""
        return r * self.N + c

    def build_hubo_terms(self):
        """
        Build all HUBO penalty terms as a dictionary:
        {frozenset of variable indices: coefficient}
        
        The constant offset is stored under frozenset().
        """
        terms = {}

        def add_term(indices, coeff):
            key = frozenset(indices)
            terms[key] = terms.get(key, 0.0) + coeff

        # --- Constraint 1: Consecutive limit ---
        # H_consec = x_i*x_j*x_k + (1-x_i)(1-x_j)(1-x_k)
        # Expanded: 1 - x_i - x_j - x_k + x_i*x_j + x_j*x_k + x_i*x_k
        # (cubic terms cancel: x_i*x_j*x_k - x_i*x_j*x_k = 0)
        for r in range(self.N):
            for c in range(self.N - 2):
                i, j, k = self.var_index(r, c), self.var_index(r, c+1), self.var_index(r, c+2)
                add_term([], self.P1)         # constant +1
                add_term([i], -self.P1)       # -x_i
                add_term([j], -self.P1)       # -x_j
                add_term([k], -self.P1)       # -x_k
                add_term([i, j], self.P1)     # +x_i*x_j
                add_term([j, k], self.P1)     # +x_j*x_k
                add_term([i, k], self.P1)     # +x_i*x_k

        for c in range(self.N):
            for r in range(self.N - 2):
                i, j, k = self.var_index(r, c), self.var_index(r+1, c), self.var_index(r+2, c)
                add_term([], self.P1)
                add_term([i], -self.P1)
                add_term([j], -self.P1)
                add_term([k], -self.P1)
                add_term([i, j], self.P1)
                add_term([j, k], self.P1)
                add_term([i, k], self.P1)

        # --- Constraint 2: Cardinality (sum = N/2 per row and column) ---
        # H = (sum_i x_i - N/2)^2 = sum_i x_i + 2*sum_{i<j} x_i*x_j - N*sum_i x_i + (N/2)^2
        # = (1 - N)*sum_i x_i + 2*sum_{i<j} x_i*x_j + (N/2)^2
        target = self.N // 2

        for r in range(self.N):
            vars_row = [self.var_index(r, c) for c in range(self.N)]
            add_term([], self.P2 * target**2)
            for v in vars_row:
                add_term([v], self.P2 * (1 - 2*target))
            for a, b in combinations(vars_row, 2):
                add_term([a, b], 2 * self.P2)

        for c in range(self.N):
            vars_col = [self.var_index(r, c) for r in range(self.N)]
            add_term([], self.P2 * target**2)
            for v in vars_col:
                add_term([v], self.P2 * (1 - 2*target))
            for a, b in combinations(vars_col, 2):
                add_term([a, b], 2 * self.P2)

        # --- Constraint 3: Uniqueness (no two rows identical, no two cols identical) ---
        # For rows r, s: penalty = prod_{c=0}^{N-1} e_{r,s,c}
        # where e_{r,s,c} = x_{r,c}*x_{s,c} + (1-x_{r,c})*(1-x_{s,c})
        #                  = 2*x_{r,c}*x_{s,c} - x_{r,c} - x_{s,c} + 1
        # The product is a polynomial of degree 2N (=8 for N=4)

        # Row uniqueness
        for r1, r2 in combinations(range(self.N), 2):
            self._add_uniqueness_product(
                [(self.var_index(r1, c), self.var_index(r2, c)) for c in range(self.N)],
                terms, add_term
            )

        # Column uniqueness
        for c1, c2 in combinations(range(self.N), 2):
            self._add_uniqueness_product(
                [(self.var_index(r, c1), self.var_index(r, c2)) for r in range(self.N)],
                terms, add_term
            )

        return terms

    def _add_uniqueness_product(self, pairs, terms, add_term):
        """
        Expand the product of XNOR terms for uniqueness constraint.
        
        Each pair (a, b) contributes factor: 2*x_a*x_b - x_a - x_b + 1
        We expand the product of N such factors into a polynomial.
        
        Representation: a polynomial is a dict {frozenset(var_indices): coeff}
        """
        # Start with polynomial = 1 (constant)
        poly = {frozenset(): 1.0}

        for (a, b) in pairs:
            # Factor: 2*x_a*x_b - x_a - x_b + 1
            factor_terms = {
                frozenset(): 1.0,
                frozenset([a]): -1.0,
                frozenset([b]): -1.0,
                frozenset([a, b]): 2.0,
            }
            # Multiply poly by factor
            new_poly = {}
            for p_vars, p_coeff in poly.items():
                for f_vars, f_coeff in factor_terms.items():
                    combined = p_vars | f_vars
                    new_poly[combined] = new_poly.get(combined, 0.0) + p_coeff * f_coeff
            poly = new_poly

        # Add all polynomial terms with penalty weight P3
        for var_set, coeff in poly.items():
            if abs(coeff) > 1e-12:
                add_term(list(var_set), self.P3 * coeff)

    def hubo_to_hamiltonian(self, terms):
        """
        Convert HUBO terms to a Qiskit SparsePauliOp (Ising Hamiltonian).
        
        Uses substitution: x_i = (I - Z_i) / 2
        
        A monomial x_{i1} * x_{i2} * ... * x_{ik} becomes:
        (1/2^k) * prod_{j=1}^k (I - Z_{ij})
        
        Expanding this product gives a sum over all subsets S of {i1,...,ik}:
        (1/2^k) * (-1)^|S| * Z_S (where Z_S = tensor product of Z on qubits in S)
        """
        pauli_terms = {}  # {pauli_string: coefficient}

        for var_set, coeff in terms.items():
            if abs(coeff) < 1e-12:
                continue
            indices = list(var_set)
            k = len(indices)
            prefactor = coeff / (2**k)

            # Enumerate all subsets of the variable set
            for mask in range(2**k):
                subset = [indices[j] for j in range(k) if (mask >> j) & 1]
                sign = (-1) ** len(subset)

                # Build Pauli string: Z on qubits in subset, I elsewhere
                pauli_str = ['I'] * self.n_qubits
                for q in subset:
                    pauli_str[q] = 'Z'
                # Qiskit uses little-endian ordering (qubit 0 is rightmost)
                pauli_label = ''.join(reversed(pauli_str))

                pauli_terms[pauli_label] = pauli_terms.get(pauli_label, 0.0) + prefactor * sign

        # Build SparsePauliOp
        labels = []
        coeffs = []
        for label, c in pauli_terms.items():
            if abs(c) > 1e-12:
                labels.append(label)
                coeffs.append(c)

        return SparsePauliOp.from_list(list(zip(labels, coeffs))).simplify()

    def evaluate_bitstring(self, bitstring, terms):
        """Evaluate the HUBO cost for a given bitstring (list/array of 0/1)."""
        energy = 0.0
        for var_set, coeff in terms.items():
            val = 1.0
            for idx in var_set:
                val *= bitstring[idx]
            energy += coeff * val
        return energy

    def bitstring_to_grid(self, bitstring):
        """Convert a bitstring to an NxN grid."""
        return np.array(bitstring).reshape(self.N, self.N)

    def validate_grid(self, grid):
        """Check if a grid satisfies all Takuzu rules."""
        N = self.N
        # Check cardinality
        for r in range(N):
            if np.sum(grid[r]) != N // 2:
                return False
        for c in range(N):
            if np.sum(grid[:, c]) != N // 2:
                return False
        # Check consecutive
        for r in range(N):
            for c in range(N - 2):
                if grid[r, c] == grid[r, c+1] == grid[r, c+2]:
                    return False
        for c in range(N):
            for r in range(N - 2):
                if grid[r, c] == grid[r+1, c] == grid[r+2, c]:
                    return False
        # Check uniqueness
        rows = [tuple(grid[r]) for r in range(N)]
        if len(set(rows)) != N:
            return False
        cols = [tuple(grid[:, c]) for c in range(N)]
        if len(set(cols)) != N:
            return False
        return True


def run_qaoa_native_hubo(N=4, p_depth=1, max_iter=200, shots=4096,
                         P1=4.0, P2=4.0, P3=6.0):
    """
    Run QAOA natively on the HUBO Hamiltonian for an NxN Takuzu grid.
    
    Args:
        N: Grid size.
        p_depth: QAOA circuit depth.
        max_iter: Max optimizer iterations.
        shots: Number of measurement shots.
        P1, P2, P3: Penalty weights.
    
    Returns:
        Dictionary with results.
    """
    import time

    print(f"=== QAOA on Native HUBO (N={N}, p={p_depth}) ===")
    print(f"Qubits: {N*N}, Penalty weights: P1={P1}, P2={P2}, P3={P3}")

    # 1. Build HUBO
    hubo = TakuzuHUBO(N=N, P1=P1, P2=P2, P3=P3)
    print("Building HUBO terms...")
    t0 = time.time()
    terms = hubo.build_hubo_terms()
    print(f"  HUBO has {len(terms)} unique terms (built in {time.time()-t0:.2f}s)")

    # 2. Convert to Hamiltonian
    print("Converting to Pauli Hamiltonian...")
    t0 = time.time()
    hamiltonian = hubo.hubo_to_hamiltonian(terms)
    print(f"  Hamiltonian has {len(hamiltonian)} Pauli terms (built in {time.time()-t0:.2f}s)")

    # 3. Run QAOA
    print(f"Running QAOA (depth p={p_depth}, max_iter={max_iter}, shots={shots})...")
    sampler = Sampler(
        backend_options={"method": "automatic"},
        run_options={"shots": shots}
    )
    optimizer = COBYLA(maxiter=max_iter)

    qaoa = QAOA(sampler=sampler, optimizer=optimizer, reps=p_depth)

    t0 = time.time()
    result = qaoa.compute_minimum_eigenvalue(hamiltonian)
    elapsed = time.time() - t0

    print(f"  Optimization completed in {elapsed:.2f}s")
    print(f"  Best eigenvalue: {result.eigenvalue:.6f}")

    # 4. Extract best bitstring from the result
    if result.best_measurement is not None:
        best_bits_str = result.best_measurement['bitstring']
        # Qiskit returns bitstrings in little-endian order
        best_bits = [int(b) for b in reversed(best_bits_str)]
    else:
        # Fallback: get from eigenstate
        best_bits = None

    # 5. Evaluate and display
    if best_bits is not None:
        grid = hubo.bitstring_to_grid(best_bits)
        energy = hubo.evaluate_bitstring(best_bits, terms)
        valid = hubo.validate_grid(grid)

        print(f"\n  Best measurement energy (HUBO): {energy:.4f}")
        print(f"  Valid Takuzu solution: {valid}")
        print(f"  Grid:\n{grid}")
    else:
        grid = None
        energy = None
        valid = False
        print("  No measurement result extracted.")

    # 6. Compute approximation ratio (energy / ground_state_energy)
    # Ground state energy for valid solution is 0 (all penalties satisfied)
    # but the constant terms shift it. Let's compute the constant.
    constant = terms.get(frozenset(), 0.0)

    results = {
        'N': N,
        'p_depth': p_depth,
        'eigenvalue': float(np.real(result.eigenvalue)),
        'best_energy': energy,
        'valid_solution': valid,
        'grid': grid,
        'elapsed_time': elapsed,
        'n_qubits': N * N,
        'n_hubo_terms': len(terms),
        'n_pauli_terms': len(hamiltonian),
        'constant_offset': constant,
    }

    print(f"\n{'='*60}\n")
    return results


def run_brute_force_verification(N=4, P1=4.0, P2=4.0, P3=6.0):
    """
    Brute-force verify the HUBO for small N by checking all 2^(N^2) states.
    Only feasible for N=4 (65536 states).
    """
    import time

    print(f"=== Brute-Force Verification (N={N}) ===")
    hubo = TakuzuHUBO(N=N, P1=P1, P2=P2, P3=P3)
    terms = hubo.build_hubo_terms()
    
    n = N * N
    best_energy = float('inf')
    best_states = []
    valid_count = 0

    t0 = time.time()
    for state_int in range(2**n):
        bits = [(state_int >> i) & 1 for i in range(n)]
        energy = hubo.evaluate_bitstring(bits, terms)
        
        if energy < best_energy - 1e-9:
            best_energy = energy
            best_states = [bits]
        elif abs(energy - best_energy) < 1e-9:
            best_states.append(bits)

        grid = hubo.bitstring_to_grid(bits)
        if hubo.validate_grid(grid):
            valid_count += 1

    elapsed = time.time() - t0
    print(f"  Enumerated {2**n} states in {elapsed:.2f}s")
    print(f"  Ground state energy: {best_energy:.6f}")
    print(f"  Number of ground states: {len(best_states)}")
    print(f"  Number of valid Takuzu solutions: {valid_count}")

    # Check if ground states are valid
    print("\n  Ground state grids:")
    for bits in best_states[:5]:  # Show up to 5
        grid = hubo.bitstring_to_grid(bits)
        valid = hubo.validate_grid(grid)
        print(f"    Valid={valid}, Grid:\n{grid}\n")

    return best_energy, best_states, valid_count


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "verify":
        run_brute_force_verification(N=4)
    else:
        # Run QAOA at different depths
        results = []
        for p in [1, 2, 3]:
            r = run_qaoa_native_hubo(N=4, p_depth=p, max_iter=300, shots=8192)
            results.append(r)

        print("\n" + "="*60)
        print("SUMMARY")
        print("="*60)
        print(f"{'Depth p':<10} {'Eigenvalue':<14} {'Valid?':<8} {'Time (s)':<10}")
        print("-"*42)
        for r in results:
            print(f"{r['p_depth']:<10} {r['eigenvalue']:<14.4f} {str(r['valid_solution']):<8} {r['elapsed_time']:<10.2f}")
