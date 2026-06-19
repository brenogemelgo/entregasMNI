import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data = {
    "x": [
        0.00,
        0.25,
        0.50,
        0.75,
        1.00,
        1.25,
        1.50,
        1.75,
        2.00,
        2.25,
        2.50,
        2.75,
        3.00,
        3.25,
        3.50,
    ],
    "y": [
        -8.00,
        -6.50,
        -2.30,
        -0.50,
        -0.20,
        0.05,
        0.10,
        0.30,
        0.40,
        0.30,
        0.50,
        0.26,
        0.90,
        3.50,
        6.50,
    ],
}

df = pd.DataFrame(data)

x = df["x"]
y = df["y"]

n = len(x)

sum_x = np.sum(x)
sum_y = np.sum(y)
sum_xx = np.sum(x * x)
sum_xy = np.sum(x * y)

# ============================================================
# REGRESSÃO LINEAR
# ============================================================

A = np.array(
    [
        [n, sum_x],
        [sum_x, sum_xx],
    ]
)

b = np.array(
    [
        sum_y,
        sum_xy,
    ]
)

sol_linear = np.linalg.solve(A, b)

a0 = sol_linear[0]
a1 = sol_linear[1]

y_linear = a0 + a1 * x

Sr_linear = np.sum((y - y_linear) ** 2)
y_mean = np.mean(y)
St = np.sum((y - y_mean) ** 2)

R2_linear = (St - Sr_linear) / St

Syx_linear = np.sqrt(Sr_linear / (n - 2))

r_linear = (n * np.sum(x * y) - np.sum(x) * np.sum(y)) / np.sqrt(
    (n * np.sum(x * x) - np.sum(x) ** 2) * (n * np.sum(y * y) - np.sum(y) ** 2)
)

print("REGRESSÃO LINEAR")
print(f"y = {a0:.6f} + {a1:.6f}x")
print(f"Erro padrão da estimativa: {Syx_linear:.6f}")
print(f"Coeficiente de correlação: {r_linear:.6f}")
print(f"R²: {R2_linear:.6f}")
print()


# ============================================================
# REGRESSÃO QUADRÁTICA
# ============================================================

sum_x2 = np.sum(x**2)
sum_x3 = np.sum(x**3)
sum_x4 = np.sum(x**4)

sum_yx0 = np.sum(y)
sum_yx1 = np.sum(y * x)
sum_yx2 = np.sum(y * x**2)

A = np.array(
    [
        [n, sum_x, sum_x2],
        [sum_x, sum_x2, sum_x3],
        [sum_x2, sum_x3, sum_x4],
    ]
)

b = np.array(
    [
        sum_yx0,
        sum_yx1,
        sum_yx2,
    ]
)

sol_quad = np.linalg.solve(A, b)

a0q = sol_quad[0]
a1q = sol_quad[1]
a2q = sol_quad[2]

y_quad = a0q + a1q * x + a2q * x**2

Sr_quad = np.sum((y - y_quad) ** 2)
R2_quad = (St - Sr_quad) / St
Syx_quad = np.sqrt(Sr_quad / (n - 3))

print("REGRESSÃO QUADRÁTICA")
print(f"y = {a0q:.6f} + {a1q:.6f}x + {a2q:.6f}x²")
print(f"Erro padrão da estimativa: {Syx_quad:.6f}")
print(f"R²: {R2_quad:.6f}")
print()


# ============================================================
# REGRESSÃO CÚBICA
# ============================================================

sum_x5 = np.sum(x**5)
sum_x6 = np.sum(x**6)

sum_yx3 = np.sum(y * x**3)

A = np.array(
    [
        [n, sum_x, sum_x2, sum_x3],
        [sum_x, sum_x2, sum_x3, sum_x4],
        [sum_x2, sum_x3, sum_x4, sum_x5],
        [sum_x3, sum_x4, sum_x5, sum_x6],
    ]
)

b = np.array(
    [
        sum_yx0,
        sum_yx1,
        sum_yx2,
        sum_yx3,
    ]
)

sol_cub = np.linalg.solve(A, b)

a0c = sol_cub[0]
a1c = sol_cub[1]
a2c = sol_cub[2]
a3c = sol_cub[3]

y_cub = a0c + a1c * x + a2c * x**2 + a3c * x**3

Sr_cub = np.sum((y - y_cub) ** 2)
R2_cub = (St - Sr_cub) / St
Syx_cub = np.sqrt(Sr_cub / (n - 4))

print("REGRESSÃO CÚBICA")
print(f"y = {a0c:.6f} + {a1c:.6f}x + {a2c:.6f}x² + {a3c:.6f}x³")
print(f"Erro padrão da estimativa: {Syx_cub:.6f}")
print(f"R²: {R2_cub:.6f}")
print()


# ============================================================
# COMPARAÇÃO
# ============================================================

print("COMPARAÇÃO")
print(
    f"Linear:     Sr = {Sr_linear:.6f}, Sy/x = {Syx_linear:.6f}, R² = {R2_linear:.6f}"
)
print(f"Quadrática: Sr = {Sr_quad:.6f}, Sy/x = {Syx_quad:.6f}, R² = {R2_quad:.6f}")
print(f"Cúbica:     Sr = {Sr_cub:.6f}, Sy/x = {Syx_cub:.6f}, R² = {R2_cub:.6f}")


# ============================================================
# GRÁFICO
# ============================================================

x_plot = np.linspace(np.min(x), np.max(x), 500)

y_plot_linear = a0 + a1 * x_plot
y_plot_quad = a0q + a1q * x_plot + a2q * x_plot**2
y_plot_cub = a0c + a1c * x_plot + a2c * x_plot**2 + a3c * x_plot**3

plt.plot(x, y, "or", label="Dados Discretos")
plt.plot(x_plot, y_plot_linear, "-b", label="Ajuste Linear")
plt.plot(x_plot, y_plot_quad, "-g", label="Ajuste Quadrático")
plt.plot(x_plot, y_plot_cub, "-k", label="Ajuste Cúbico")

plt.legend()
plt.xlabel("$x$")
plt.ylabel("$y$")
plt.grid(True)

plt.show()
