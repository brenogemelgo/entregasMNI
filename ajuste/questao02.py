import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# valores de x e y gerados sinteticamente usando o arquivo pseudo.py.
# curva:
# tau = Ky^n
# valores:
# K = 0.958027
# n = 0.355695
data = {
    "ydot": [10, 20, 30, 40, 50, 70, 90, 110, 130, 150, 170, 200],
    "tau": [2.19, 2.82, 3.2, 3.63, 3.88, 4.4, 4.85, 4.96, 5.26, 5.59, 5.86, 6.38],
}

df = pd.DataFrame(data)

x = df["ydot"]
y = df["tau"]

n = len(x)

y_mean = np.mean(y)
St = np.sum((y - y_mean) ** 2)

# ============================================================
# MODELO EXPONENCIAL
# ============================================================

Y = np.log(y)

sum_x = np.sum(x)
sum_Y = np.sum(Y)
sum_xx = np.sum(x * x)
sum_Yx = np.sum(Y * x)

A = np.array(
    [
        [n, sum_x],
        [sum_x, sum_xx],
    ]
)

b = np.array(
    [
        sum_Y,
        sum_Yx,
    ]
)

sol_exp = np.linalg.solve(A, b)

ln_a_exp = sol_exp[0]
b_exp = sol_exp[1]

a_exp = np.exp(ln_a_exp)

y_exp = a_exp * np.exp(b_exp * x)

Sr_exp = np.sum((y - y_exp) ** 2)
Syx_exp = np.sqrt(Sr_exp / (n - 2))
R2_exp = (St - Sr_exp) / St

print("MODELO EXPONENCIAL")
print(f"y = {a_exp:.6f} exp({b_exp:.6f}x)")
print(f"Sr = {Sr_exp:.6f}")
print(f"Sy/x = {Syx_exp:.6f}")
print(f"R² = {R2_exp:.6f}")
print()


# ============================================================
# MODELO DE POTÊNCIA SIMPLES
# ============================================================

X = np.log10(x)
Y = np.log10(y)

sum_X = np.sum(X)
sum_Y = np.sum(Y)
sum_XX = np.sum(X * X)
sum_YX = np.sum(Y * X)

A = np.array(
    [
        [n, sum_X],
        [sum_X, sum_XX],
    ]
)

b = np.array(
    [
        sum_Y,
        sum_YX,
    ]
)

sol_pot = np.linalg.solve(A, b)

log10_a_pot = sol_pot[0]
b_pot = sol_pot[1]

a_pot = 10**log10_a_pot

y_pot = a_pot * x**b_pot

Sr_pot = np.sum((y - y_pot) ** 2)
Syx_pot = np.sqrt(Sr_pot / (n - 2))
R2_pot = (St - Sr_pot) / St

print("MODELO DE POTÊNCIA SIMPLES")
print(f"y = {a_pot:.6f} x^{b_pot:.6f}")
print(f"Sr = {Sr_pot:.6f}")
print(f"Sy/x = {Syx_pot:.6f}")
print(f"R² = {R2_pot:.6f}")
print()


# ============================================================
# MODELO DE TAXA DE SATURAÇÃO
# ============================================================

X = 1 / x
Y = 1 / y

sum_X = np.sum(X)
sum_Y = np.sum(Y)
sum_XX = np.sum(X * X)
sum_YX = np.sum(Y * X)

A = np.array(
    [
        [n, sum_X],
        [sum_X, sum_XX],
    ]
)

b = np.array(
    [
        sum_Y,
        sum_YX,
    ]
)

sol_sat = np.linalg.solve(A, b)

c0 = sol_sat[0]
c1 = sol_sat[1]

a_sat = 1 / c0
b_sat = c1 / c0

y_sat = a_sat * x / (b_sat + x)

Sr_sat = np.sum((y - y_sat) ** 2)
Syx_sat = np.sqrt(Sr_sat / (n - 2))
R2_sat = (St - Sr_sat) / St

print("MODELO DE TAXA DE SATURAÇÃO")
print(f"y = {a_sat:.6f}x / ({b_sat:.6f} + x)")
print(f"Sr = {Sr_sat:.6f}")
print(f"Sy/x = {Syx_sat:.6f}")
print(f"R² = {R2_sat:.6f}")
print()


# ============================================================
# REGRESSÃO QUADRÁTICA / PARÁBOLA
# ============================================================

sum_x = np.sum(x)
sum_y = np.sum(y)
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

sol_par = np.linalg.solve(A, b)

a0_par = sol_par[0]
a1_par = sol_par[1]
a2_par = sol_par[2]

y_par = a0_par + a1_par * x + a2_par * x**2

Sr_par = np.sum((y - y_par) ** 2)
Syx_par = np.sqrt(Sr_par / (n - 3))
R2_par = (St - Sr_par) / St

print("PARÁBOLA")
print(f"y = {a0_par:.6f} + {a1_par:.6f}x + {a2_par:.6f}x²")
print(f"Sr = {Sr_par:.6f}")
print(f"Sy/x = {Syx_par:.6f}")
print(f"R² = {R2_par:.6f}")
print()


# ============================================================
# COMPARAÇÃO
# ============================================================

print("COMPARAÇÃO")
print(f"Exponencial:        Sr = {Sr_exp:.6f}, Sy/x = {Syx_exp:.6f}, R² = {R2_exp:.6f}")
print(f"Potência simples:   Sr = {Sr_pot:.6f}, Sy/x = {Syx_pot:.6f}, R² = {R2_pot:.6f}")
print(f"Taxa de saturação:  Sr = {Sr_sat:.6f}, Sy/x = {Syx_sat:.6f}, R² = {R2_sat:.6f}")
print(f"Parábola:           Sr = {Sr_par:.6f}, Sy/x = {Syx_par:.6f}, R² = {R2_par:.6f}")


# ============================================================
# GRÁFICO
# ============================================================

x_plot = np.linspace(np.min(x), np.max(x), 500)

y_plot_exp = a_exp * np.exp(b_exp * x_plot)
y_plot_pot = a_pot * x_plot**b_pot
y_plot_sat = a_sat * x_plot / (b_sat + x_plot)
y_plot_par = a0_par + a1_par * x_plot + a2_par * x_plot**2

plt.plot(x, y, "or", label="Dados discretos")
plt.plot(x_plot, y_plot_exp, "-b", label="Exponencial")
plt.plot(x_plot, y_plot_pot, "-g", label="Potência simples")
plt.plot(x_plot, y_plot_sat, "-k", label="Taxa de saturação")
plt.plot(x_plot, y_plot_par, "--m", label="Parábola")

plt.legend()
plt.xlabel(r"$\dot{\gamma}$ [$\mathrm{s^{-1}}$]")
plt.ylabel(r"$\tau$ [$\mathrm{Pa}$]")
plt.grid(True)

plt.show()
