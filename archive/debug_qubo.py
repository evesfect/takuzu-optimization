"""Diagnostic: check the QUBO matrix values and energies for known configurations."""
import numpy as np
from takuzu_2d_qubo import Takuzu2DBuilder
from qiskit_optimization.converters import QuadraticProgramToQubo

N = 4
builder = Takuzu2DBuilder(N=N)
qp = builder.build()

# Check the raw Qiskit objective before QuadraticProgramToQubo
print("=== Raw QP from builder ===")
print(f"Num variables: {qp.get_num_vars()}")
print(f"Num constraints: {qp.get_num_linear_constraints()}")
print(f"Objective constant: {qp.objective.constant}")

lin = qp.objective.linear.to_dict(use_name=True)
quad = qp.objective.quadratic.to_dict(use_name=True)
print(f"Num linear terms: {len(lin)}")
print(f"Num quadratic terms: {len(quad)}")

# Print a few linear terms
for k, v in sorted(lin.items())[:8]:
    print(f"  lin[{k}] = {v}")

# Check accumulated dicts from builder
print(f"\n=== Builder accumulators ===")
print(f"lin_dict entries: {len(builder.lin_dict)}")
print(f"quad_dict entries: {len(builder.quad_dict)}")
print(f"const: {builder.const}")

# Now convert to pure QUBO (same as test script)
conv = QuadraticProgramToQubo()
qubo_problem = conv.convert(qp)

print(f"\n=== After QuadraticProgramToQubo ===")
print(f"Num variables: {qubo_problem.get_num_vars()}")
print(f"Num constraints: {qubo_problem.get_num_linear_constraints()}")
print(f"Objective constant: {qubo_problem.objective.constant}")

lin2 = qubo_problem.objective.linear.to_dict(use_name=True)
quad2 = qubo_problem.objective.quadratic.to_dict(use_name=True)
print(f"Num linear terms: {len(lin2)}")
print(f"Num quadratic terms: {len(quad2)}")

# Build Q matrix same way as test script
linear_idx = qubo_problem.objective.linear.to_dict()
quadratic_idx = qubo_problem.objective.quadratic.to_dict()
var_names = [v.name for v in qubo_problem.variables]

Q = {}
for i, val in linear_idx.items():
    Q[(var_names[i], var_names[i])] = val
for (i, j), val in quadratic_idx.items():
    Q[(var_names[i], var_names[j])] = val

print(f"\n=== Q matrix ===")
print(f"Total entries: {len(Q)}")
print(f"Sum of all Q values: {sum(Q.values())}")

# Check Q entries for x variables only
x_entries = {k: v for k, v in Q.items() if k[0].startswith('x') and k[1].startswith('x')}
print(f"Q entries involving only x vars: {len(x_entries)}")
print(f"Sum of x-only Q values: {sum(x_entries.values())}")

# Evaluate energy for all-zeros
def eval_energy(Q, sample):
    e = 0
    for (i, j), val in Q.items():
        e += val * sample.get(i, 0) * sample.get(j, 0)
    return e

all_zeros = {v.name: 0 for v in qubo_problem.variables}
all_ones_x = {v.name: 0 for v in qubo_problem.variables}
for r in range(N):
    for c in range(N):
        all_ones_x[f'x_{r}_{c}'] = 1

# A known valid Takuzu board
valid_board = [
    [1, 0, 1, 0],
    [0, 1, 0, 1],
    [1, 0, 0, 1],
    [0, 1, 1, 0],
]
valid_sample = {v.name: 0 for v in qubo_problem.variables}
for r in range(N):
    for c in range(N):
        valid_sample[f'x_{r}_{c}'] = valid_board[r][c]

# For valid sample, set w variables optimally (w = xi*xj)
for r in range(N):
    for c in range(N-2):
        w_name = f'w_row_{r}_col_{c}'
        xi = valid_board[r][c]
        xj = valid_board[r][c+1]
        valid_sample[w_name] = xi * xj

for c_idx in range(N):
    for r in range(N-2):
        w_name = f'w_col_{c_idx}_row_{r}'
        xi = valid_board[r][c_idx]
        xj = valid_board[r+1][c_idx]
        valid_sample[w_name] = xi * xj

print(f"\n=== Energy evaluations (Q only, no constant) ===")
print(f"All zeros: {eval_energy(Q, all_zeros)}")
print(f"All x=1:   {eval_energy(Q, all_ones_x)}")
print(f"Valid board: {eval_energy(Q, valid_sample)}")
print(f"\nWith constant ({qubo_problem.objective.constant}):")
c = qubo_problem.objective.constant
print(f"All zeros: {eval_energy(Q, all_zeros) + c}")
print(f"All x=1:   {eval_energy(Q, all_ones_x) + c}")
print(f"Valid board: {eval_energy(Q, valid_sample) + c}")
