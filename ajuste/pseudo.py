import numpy as np

rng = np.random.default_rng(seed=None)

x = np.array([10, 20, 30, 40, 50, 70, 90, 110, 130, 150, 170, 200], dtype=float)

K = rng.uniform(0.60, 1.30)
n = rng.uniform(0.30, 0.65)

y_base = K * x**n

perturbacao = rng.uniform(-0.03, 0.03, size=len(x))

y = y_base * (1.0 + perturbacao)
y = np.round(y, 2)

data = {
    "x": x.astype(int).tolist(),
    "y": y.tolist(),
}

print(f"K = {K:.6f}")
print(f"n = {n:.6f}")
print(data)
