import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data = {
    "x": [0.0, 1.8, 5.0, 6.0, 8.2, 9.2, 12.0],
    "y": [26.000, 16.415, 5.375, 3.500, 2.015, 2.540, 8.000],
}

df = pd.DataFrame(data)

x = df["x"].to_numpy(dtype=float)
y = df["y"].to_numpy(dtype=float)

x_alvo = 3.5
n = len(x)

# ============================================================
# TABELA DE DIFERENÇAS DIVIDIDAS DE NEWTON
# ============================================================

dd = np.zeros((n, n))

for i in range(n):
    dd[i, 0] = y[i]

for j in range(1, n):
    for i in range(n - j):
        dd[i, j] = (dd[i + 1, j - 1] - dd[i, j - 1]) / (x[i + j] - x[i])

b = dd[0, :]

print("COEFICIENTES DE NEWTON")
for i in range(n):
    print(f"b{i} = {b[i]:.10f}")

print()


# ============================================================
# IDENTIFICAÇÃO DO MENOR GRAU EXATO
# ============================================================

criterio = 1e-16
grau_exato = 0

for i in range(n):
    if abs(b[i]) > criterio:
        grau_exato = i

print("GRAU IDENTIFICADO")
print(f"grau = {grau_exato}")
print()


# ============================================================
# AVALIAÇÃO DO POLINÔMIO DE NEWTON
# ============================================================


def newton(x_alvo, x, b, grau):
    soma = b[0]
    produto = 1.0

    for j in range(1, grau + 1):
        produto *= x_alvo - x[j - 1]
        soma += b[j] * produto

    return soma


y_alvo = newton(x_alvo, x, b, grau_exato)

print("VALOR INTERPOLADO")
print(f"f(3.5) = {y_alvo:.10f}")
print()


# ============================================================
# VERIFICAÇÃO DO ERRO NOS PONTOS DA TABELA
# ============================================================

erros = []

for i in range(n):
    y_calc = newton(x[i], x, b, grau_exato)
    erro = y[i] - y_calc
    erros.append(abs(erro))

erro_max = np.max(erros)

print("ERRO NOS PONTOS DA TABELA")
print(f"erro máximo = {erro_max:.10e}")
print()

print("VALORES CALCULADOS NOS PONTOS")
for i in range(n):
    y_calc = newton(x[i], x, b, grau_exato)

    print(
        f"x = {x[i]:5.1f} | "
        f"y tabela = {y[i]:8.3f} | "
        f"f(x) = {y_calc:8.3f} | "
        f"erro = {y[i] - y_calc: .3e}"
    )

print()


# ============================================================
# POLINÔMIO EXPANDIDO
# ============================================================

A = np.array(
    [
        [1, x[0], x[0] ** 2],
        [1, x[1], x[1] ** 2],
        [1, x[2], x[2] ** 2],
    ]
)

bb = np.array(
    [
        y[0],
        y[1],
        y[2],
    ]
)

sol = np.linalg.solve(A, bb)

a1 = sol[0]
a2 = sol[1]
a3 = sol[2]

print("POLINÔMIO EXPANDIDO")
print(f"f2(x) = {a1:.6f} + ({a2:.6f})x + ({a3:.6f})x²")
print()


# ============================================================
# GRÁFICO DOS PONTOS VS POLINÔMIO ENCONTRADO
# ============================================================

x_plot = np.linspace(np.min(x), np.max(x), 500)
y_plot = a1 + a2 * x_plot + a3 * x_plot**2

plt.plot(x, y, "or", label="Dados da tabela")
plt.plot(x_plot, y_plot, "-b", label=r"$f_2(x)=26-6x+0.375x^2$")

plt.xlabel("$x$")
plt.ylabel("$y$")
plt.grid(True)
plt.legend()

plt.show()
