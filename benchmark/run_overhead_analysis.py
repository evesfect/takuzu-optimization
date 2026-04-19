from takuzu_model import TakuzuQuboGenerator

def run_overhead_analysis():
    grids = [4, 6, 8]
    
    print(f"{'Grid Size':<12} | {'Primary Vars':<14} | {'Aux Vars (Base)':<18} | {'Aux Vars (w/ Uniqueness)':<25}")
    print("-" * 75)
    
    for N in grids:
        # 1. Count Base Rules (No Uniqueness)
        builder_base = TakuzuQuboGenerator(N=N, enforce_uniqueness=False)
        Q_base, _ = builder_base.build()
        primary_vars = N * N
        total_vars_base = len(set([k[0] for k in Q_base.keys()]))
        aux_base = total_vars_base - primary_vars
        
        # 2. Count Full Rules (With Uniqueness)
        builder_full = TakuzuQuboGenerator(N=N, enforce_uniqueness=True)
        Q_full, _ = builder_full.build()
        total_vars_full = len(set([k[0] for k in Q_full.keys()]))
        aux_full = total_vars_full - primary_vars
        
        print(f"{N}x{N:<10} | {primary_vars:<14} | {aux_base:<18} | {aux_full:<25}")

if __name__ == "__main__":
    run_overhead_analysis()