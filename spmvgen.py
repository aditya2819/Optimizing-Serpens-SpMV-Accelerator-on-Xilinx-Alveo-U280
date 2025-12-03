import random

sizes = [
    (50_000,   50_000),
    (100_000, 100_000),
    (500_000, 500_000),
    (1_000_000, 1_000_000)
]

def nnzPerColumn(n):
    if n <= 50_000:
        return 70
    elif n <= 100_000:
        return 80
    elif n <= 500_000:
        return 90
    elif n <= 1_000_000:
        return 100

for (rows, cols) in sizes:
    N = rows
    nnzCol = nnzPerColumn(N)
    totalNnz = N * nnzCol
    outname = f"matrix_{N}_{N}_{totalNnz}.mtx"
    print("Generating:", outname)
    print("  Dimension:", N, "x", N)
    print("  NNZ per column:", nnzCol)
    print("  Total NNZ:", totalNnz)
    with open(outname, "w") as f:
        f.write("%%MatrixMarket matrix coordinate real general\n")
        f.write(f"{N} {N} {totalNnz}\n")
        for col in range(1, N + 1):
            rows = random.sample(range(1, N + 1), nnzCol)
            for r in rows:
                f.write(f"{r} {col} 1.0\n")
    print("  Done.\n")
print("All matrices generated.")