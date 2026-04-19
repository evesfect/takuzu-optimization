from pysat.examples.rc2 import RC2
from pysat.formula import WCNF

def test_consecutive_rule():
    # Initialize a Weighted CNF formula
    wcnf = WCNF()
    
    # Let's model a 4-cell row: variables 1, 2, 3, 4
    # We have two consecutive triplets: (1,2,3) and (2,3,4)
    
    # Triplet 1: (1, 2, 3)
    # Clause 1: Not all 1s -> (Not 1 OR Not 2 OR Not 3)
    wcnf.append([-1, -2, -3], weight=1)
    # Clause 2: Not all 0s -> (1 OR 2 OR 3)
    wcnf.append([1, 2, 3], weight=1)
    
    # Triplet 2: (2, 3, 4)
    wcnf.append([-2, -3, -4], weight=1)
    wcnf.append([2, 3, 4], weight=1)

    # Solve using RC2 Max-SAT solver
    with RC2(wcnf) as rc2:
        model = rc2.compute()
        cost = rc2.cost
        
    print(f"Optimal State Found: {model}")
    print(f"Number of Violated Clauses (Cost): {cost}")

if __name__ == "__main__":
    print("Testing PySAT RC2 Solver...")
    test_consecutive_rule()