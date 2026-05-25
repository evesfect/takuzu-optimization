"""
Classical RC2 Max-SAT solver for Takuzu.

Encodes Takuzu constraints directly as CNF clauses (no QUBO/HUBO needed).
Uses PySAT's RC2 core-guided solver for exact solutions.
"""

import time
import numpy as np
from pysat.formula import WCNF, CNF
from pysat.solvers import Solver
from pysat.card import CardEnc, EncType
from itertools import combinations


def var_id(r, c, N):
    """Map grid position (r, c) to SAT variable (1-indexed)."""
    return r * N + c + 1


def solve_takuzu_sat(N=4):
    """
    Solve an NxN Takuzu puzzle using a SAT solver with CNF encoding.
    
    Returns:
        grid: NxN numpy array with the solution.
        elapsed: Time in seconds.
    """
    print(f"=== RC2 SAT Solver (N={N}) ===")

    cnf = CNF()
    top_var = N * N  # track the highest variable used

    # --- Constraint 1: Consecutive limit ---
    # No three consecutive identical values in any row or column
    # For each triplet (i, j, k):
    #   Not all 1: (-i OR -j OR -k)
    #   Not all 0: (i OR j OR k)
    
    for r in range(N):
        for c in range(N - 2):
            i = var_id(r, c, N)
            j = var_id(r, c + 1, N)
            k = var_id(r, c + 2, N)
            cnf.append([-i, -j, -k])
            cnf.append([i, j, k])

    for c in range(N):
        for r in range(N - 2):
            i = var_id(r, c, N)
            j = var_id(r + 1, c, N)
            k = var_id(r + 2, c, N)
            cnf.append([-i, -j, -k])
            cnf.append([i, j, k])

    # --- Constraint 2: Cardinality (exactly N/2 ones per row and column) ---
    # Use sequential counter encoding for exactly-k constraints
    target = N // 2

    for r in range(N):
        lits = [var_id(r, c, N) for c in range(N)]
        # Encode: exactly target ones among lits
        # AtMost target: sum <= target
        atmost = CardEnc.atmost(lits, bound=target, top_id=top_var, encoding=EncType.seqcounter)
        top_var = atmost.nv
        for cl in atmost.clauses:
            cnf.append(cl)
        # AtLeast target: sum >= target (equiv: at most N-target negations)
        neg_lits = [-l for l in lits]
        atleast = CardEnc.atmost(neg_lits, bound=N - target, top_id=top_var, encoding=EncType.seqcounter)
        top_var = atleast.nv
        for cl in atleast.clauses:
            cnf.append(cl)

    for c in range(N):
        lits = [var_id(r, c, N) for r in range(N)]
        atmost = CardEnc.atmost(lits, bound=target, top_id=top_var, encoding=EncType.seqcounter)
        top_var = atmost.nv
        for cl in atmost.clauses:
            cnf.append(cl)
        neg_lits = [-l for l in lits]
        atleast = CardEnc.atmost(neg_lits, bound=N - target, top_id=top_var, encoding=EncType.seqcounter)
        top_var = atleast.nv
        for cl in atleast.clauses:
            cnf.append(cl)

    # --- Constraint 3: Uniqueness (no two rows identical, no two columns identical) ---
    # For each pair of rows (r1, r2): at least one column must differ
    # Encoding: for each pair, introduce N "difference indicator" variables
    # d_c = (x_{r1,c} XOR x_{r2,c}), then require OR of all d_c

    for r1, r2 in combinations(range(N), 2):
        diff_vars = []
        for c in range(N):
            v1 = var_id(r1, c, N)
            v2 = var_id(r2, c, N)
            # d = v1 XOR v2: true if they differ
            top_var += 1
            d = top_var
            diff_vars.append(d)
            # Encode d <-> (v1 XOR v2):
            # d -> (v1 OR v2) and d -> (-v1 OR -v2) [d implies exactly one]
            # -d -> (v1 OR -v2) or (-v1 OR v2) [not d implies same]
            # Standard XOR encoding:
            # (v1 OR v2 OR -d), (-v1 OR -v2 OR -d), (v1 OR -v2 OR d), (-v1 OR v2 OR d)
            cnf.append([v1, v2, -d])
            cnf.append([-v1, -v2, -d])
            cnf.append([v1, -v2, d])
            cnf.append([-v1, v2, d])
        # At least one must differ
        cnf.append(diff_vars)

    for c1, c2 in combinations(range(N), 2):
        diff_vars = []
        for r in range(N):
            v1 = var_id(r, c1, N)
            v2 = var_id(r, c2, N)
            top_var += 1
            d = top_var
            diff_vars.append(d)
            cnf.append([v1, v2, -d])
            cnf.append([-v1, -v2, -d])
            cnf.append([v1, -v2, d])
            cnf.append([-v1, v2, d])
        cnf.append(diff_vars)

    print(f"  CNF has {len(cnf.clauses)} clauses, {top_var} variables")
    print(f"  (Primary: {N*N}, Auxiliary: {top_var - N*N})")

    # Solve
    t0 = time.time()
    with Solver(name='g3', bootstrap_with=cnf) as solver:
        sat = solver.solve()
        elapsed = time.time() - t0
        if sat:
            model = solver.get_model()
        else:
            model = None

    if model is None:
        print(f"  UNSATISFIABLE (elapsed: {elapsed:.4f}s)")
        return None, elapsed

    # Extract grid from model
    grid = np.zeros((N, N), dtype=int)
    for r in range(N):
        for c in range(N):
            vid = var_id(r, c, N)
            if vid in model:
                grid[r, c] = 1
            elif -vid in model:
                grid[r, c] = 0
            else:
                # Variable not set, default to checking sign
                grid[r, c] = 1 if model[vid - 1] > 0 else 0

    print(f"  Solution found in {elapsed:.6f}s")
    print(f"  Grid:\n{grid}")

    # Validate
    valid = validate_takuzu(grid)
    print(f"  Valid: {valid}")

    return grid, elapsed


def validate_takuzu(grid):
    """Validate all Takuzu rules."""
    N = grid.shape[0]
    # Cardinality
    for r in range(N):
        if np.sum(grid[r]) != N // 2:
            return False
    for c in range(N):
        if np.sum(grid[:, c]) != N // 2:
            return False
    # Consecutive
    for r in range(N):
        for c in range(N - 2):
            if grid[r, c] == grid[r, c+1] == grid[r, c+2]:
                return False
    for c in range(N):
        for r in range(N - 2):
            if grid[r, c] == grid[r+1, c] == grid[r+2, c]:
                return False
    # Uniqueness
    rows = [tuple(grid[r]) for r in range(N)]
    if len(set(rows)) != N:
        return False
    cols = [tuple(grid[:, c]) for c in range(N)]
    if len(set(cols)) != N:
        return False
    return True


def count_all_solutions(N=4):
    """Count all valid Takuzu solutions for an NxN grid."""
    from pysat.solvers import Solver as PySATSolver

    print(f"\n=== Counting all solutions (N={N}) ===")

    cnf = CNF()
    top_var = N * N

    # Same encoding as above
    for r in range(N):
        for c in range(N - 2):
            i, j, k = var_id(r, c, N), var_id(r, c+1, N), var_id(r, c+2, N)
            cnf.append([-i, -j, -k])
            cnf.append([i, j, k])
    for c in range(N):
        for r in range(N - 2):
            i, j, k = var_id(r, c, N), var_id(r+1, c, N), var_id(r+2, c, N)
            cnf.append([-i, -j, -k])
            cnf.append([i, j, k])

    target = N // 2
    for r in range(N):
        lits = [var_id(r, c, N) for c in range(N)]
        atmost = CardEnc.atmost(lits, bound=target, top_id=top_var, encoding=EncType.seqcounter)
        top_var = atmost.nv
        for cl in atmost.clauses:
            cnf.append(cl)
        neg_lits = [-l for l in lits]
        atleast = CardEnc.atmost(neg_lits, bound=N - target, top_id=top_var, encoding=EncType.seqcounter)
        top_var = atleast.nv
        for cl in atleast.clauses:
            cnf.append(cl)
    for c in range(N):
        lits = [var_id(r, c, N) for r in range(N)]
        atmost = CardEnc.atmost(lits, bound=target, top_id=top_var, encoding=EncType.seqcounter)
        top_var = atmost.nv
        for cl in atmost.clauses:
            cnf.append(cl)
        neg_lits = [-l for l in lits]
        atleast = CardEnc.atmost(neg_lits, bound=N - target, top_id=top_var, encoding=EncType.seqcounter)
        top_var = atleast.nv
        for cl in atleast.clauses:
            cnf.append(cl)

    for r1, r2 in combinations(range(N), 2):
        diff_vars = []
        for c in range(N):
            v1, v2 = var_id(r1, c, N), var_id(r2, c, N)
            top_var += 1
            d = top_var
            diff_vars.append(d)
            cnf.append([v1, v2, -d])
            cnf.append([-v1, -v2, -d])
            cnf.append([v1, -v2, d])
            cnf.append([-v1, v2, d])
        cnf.append(diff_vars)

    for c1, c2 in combinations(range(N), 2):
        diff_vars = []
        for r in range(N):
            v1, v2 = var_id(r, c1, N), var_id(r, c2, N)
            top_var += 1
            d = top_var
            diff_vars.append(d)
            cnf.append([v1, v2, -d])
            cnf.append([-v1, -v2, -d])
            cnf.append([v1, -v2, d])
            cnf.append([-v1, v2, d])
        cnf.append(diff_vars)

    # Enumerate solutions
    solutions = []
    primary_vars = list(range(1, N*N + 1))
    t0 = time.time()

    with PySATSolver(name='g3', bootstrap_with=cnf) as solver:
        while solver.solve():
            model = solver.get_model()
            # Extract primary variable assignment
            assignment = []
            for v in primary_vars:
                if v in model:
                    assignment.append(1)
                else:
                    assignment.append(0)
            solutions.append(assignment)
            # Block this solution
            blocking = [-model[v-1] for v in primary_vars]
            solver.add_clause(blocking)

    elapsed = time.time() - t0
    print(f"  Found {len(solutions)} valid solutions in {elapsed:.4f}s")
    return solutions


if __name__ == "__main__":
    # Solve a 4x4 Takuzu
    grid, elapsed = solve_takuzu_sat(N=4)
    
    # Count all solutions
    solutions = count_all_solutions(N=4)
    
    print(f"\n  Total valid 4x4 Takuzu solutions: {len(solutions)}")
