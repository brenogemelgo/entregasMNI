import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data = {
    "x": [1.0, 2.0, 2.5, 3.0, 4.0, 5.0],
    "y": [0.0, 5.0, 6.5, 7.0, 3.0, 1.0],
}

df = pd.DataFrame(data)

x = df["x"].to_numpy(dtype=float)
y = df["y"].to_numpy(dtype=float)

x_alvo = 3.4


# ============================================================
# SEQUÊNCIA DE PONTOS ESCOLHIDA
# ============================================================

sequencias = {
    1: {
        "x": np.array([3.0, 4.0]),
        "y": np.array([7.0, 3.0]),
    },
    2: {
        "x": np.array([3.0, 4.0, 2.5]),
        "y": np.array([7.0, 3.0, 6.5]),
    },
    3: {
        "x": np.array([3.0, 4.0, 2.5, 2.0]),
        "y": np.array([7.0, 3.0, 6.5, 5.0]),
    },
}


# ============================================================
# INTERPOLAÇÃO DE NEWTON POR DIFERENÇAS DIVIDIDAS
# ============================================================


def coeficientes_newton(xp, yp):
    n = len(xp)

    dd = np.zeros((n, n))

    for i in range(n):
        dd[i, 0] = yp[i]

    for j in range(1, n):
        for i in range(n - j):
            dd[i, j] = (dd[i + 1, j - 1] - dd[i, j - 1]) / (xp[i + j] - xp[i])

    b = dd[0, :]

    return b, dd


def newton(x_alvo, xp, b):
    n = len(b)

    soma = b[0]
    produto = 1.0

    for j in range(1, n):
        produto *= x_alvo - xp[j - 1]
        soma += b[j] * produto

    return soma


# ============================================================
# INTERPOLAÇÃO DE LAGRANGE
# ============================================================


def lagrange(x_alvo, xp, yp):
    n = len(xp)

    soma = 0.0

    for i in range(n):
        L = 1.0

        for j in range(n):
            if i != j:
                L *= (x_alvo - xp[j]) / (xp[i] - xp[j])

        soma += yp[i] * L

    return soma


# ============================================================
# POLINÔMIO EXPANDIDO
# ============================================================


def polinomio_expandido(xp, yp):
    grau = len(xp) - 1

    A = np.zeros((len(xp), len(xp)))

    for i in range(len(xp)):
        for j in range(len(xp)):
            A[i, j] = xp[i] ** j

    coef = np.linalg.solve(A, yp)

    return coef


def formatar_polinomio(coef, nome="P"):
    termos = []

    for i, ai in enumerate(coef):
        if abs(ai) < 1e-12:
            continue

        if i == 0:
            termo = f"{ai:.6f}"
        elif i == 1:
            termo = f"{ai:+.6f}x"
        else:
            termo = f"{ai:+.6f}x^{i}"

        termos.append(termo)

    polinomio = " ".join(termos)

    return f"{nome}(x) = {polinomio}"


# ============================================================
# CÁLCULO DOS RESULTADOS
# ============================================================

resultados = []

for grau in [1, 2, 3]:
    xp = sequencias[grau]["x"]
    yp = sequencias[grau]["y"]

    b, dd = coeficientes_newton(xp, yp)

    f_newton = newton(x_alvo, xp, b)
    f_lagrange = lagrange(x_alvo, xp, yp)

    coef_exp = polinomio_expandido(xp, yp)

    resultados.append(
        {
            "grau": grau,
            "x_pontos": xp,
            "y_pontos": yp,
            "b": b,
            "coef_exp": coef_exp,
            "f_newton": f_newton,
            "f_lagrange": f_lagrange,
        }
    )


# ============================================================
# SAÍDA DOS RESULTADOS
# ============================================================

print("RESULTADOS")
print()

for resultado in resultados:
    grau = resultado["grau"]
    xp = resultado["x_pontos"]
    yp = resultado["y_pontos"]
    b = resultado["b"]
    coef_exp = resultado["coef_exp"]
    f_newton = resultado["f_newton"]
    f_lagrange = resultado["f_lagrange"]

    print(f"GRAU {grau}")
    print(f"x usados = {xp}")
    print(f"y usados = {yp}")

    print("Coeficientes de Newton:")
    for i in range(len(b)):
        print(f"b{i} = {b[i]:.10f}")

    print("Polinômio expandido:")
    print(formatar_polinomio(coef_exp, nome=f"P{grau}"))

    print(f"Newton:   f(3.4) = {f_newton:.10f}")
    print(f"Lagrange: f(3.4) = {f_lagrange:.10f}")
    print()


# ============================================================
# GRÁFICO DOS DADOS E POLINÔMIOS DE NEWTON
# ============================================================

x_plot = np.linspace(np.min(x), np.max(x), 500)

plt.plot(x, y, "or", label="Dados da tabela")
plt.plot(x_alvo, resultados[-1]["f_newton"], "ok", label=r"$f(3.4)$")

for resultado in resultados:
    grau = resultado["grau"]
    xp = resultado["x_pontos"]
    b = resultado["b"]

    y_plot = []

    for xi in x_plot:
        y_plot.append(newton(xi, xp, b))

    y_plot = np.array(y_plot)

    plt.plot(x_plot, y_plot, label=f"Newton grau {grau}")

plt.xlabel("$x$")
plt.ylabel("$y$")
plt.grid(True)
plt.legend()

plt.show()
