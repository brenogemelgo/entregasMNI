# ====================================================================================
#  Raízes de Funções - Questão 1
#
#  Disciplina: Métodos Numéricos I
#  Curso: Engenharia de Petróleo
#  Instituição: UDESC
#  Semestre: 2026/1
#
#  Autores:
#  - Ana Júlia Gonsalves
#  - Breno Gemelgo
#  - Odessa Rublic
# ====================================================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# =============================================================================== #
# GLOBAL


def f(x):
    return np.sin(4 * x) + np.cos(5 * x) + 1 / x


def dfdx(x):
    return 4 * np.cos(4 * x) - 5 * np.sin(5 * x) - 1 / (x**2)


def bisseccao(xl, xu, Eppara):
    Epest = 100
    xr_old = 0
    k = 0

    while Epest >= Eppara:
        xr = (xl + xu) / 2

        if (f(xl) * f(xr)) < 0:
            xu = xr
        else:
            xl = xr

        if k > 0:
            Epest = abs((xr - xr_old) / xr) * 100

        xr_old = xr
        k += 1

    return xr, k, Epest


def falsa_posicao(xl, xu, Eppara):
    Epest = 100
    xr_old = 0
    k = 0

    while Epest >= Eppara:
        xr = xu - f(xu) * (xl - xu) / (f(xl) - f(xu))

        if (f(xl) * f(xr)) < 0:
            xu = xr
        else:
            xl = xr

        if k > 0:
            Epest = abs((xr - xr_old) / xr) * 100

        xr_old = xr
        k += 1

    return xr, k, Epest


def newton_raphson(x_i, Eppara):
    Epest = 100
    xr_old = x_i
    xr = x_i
    k = 0

    while Epest >= Eppara:
        xr = xr_old - f(xr_old) / dfdx(xr_old)

        if k > 0:
            Epest = abs((xr - xr_old) / xr) * 100

        xr_old = xr
        k += 1

    return xr, k, Epest


def secante(x_im1, x_i, Eppara):
    Epest = 100
    k = 0

    while Epest >= Eppara:
        xr = x_i - f(x_i) * (x_im1 - x_i) / (f(x_im1) - f(x_i))

        if k > 0:
            Epest = abs((xr - x_i) / xr) * 100

        x_im1 = x_i
        x_i = xr
        k += 1

    return xr, k, Epest


def secante_modificada(x_i, Eppara, delta):
    Epest = 100
    xr_old = x_i
    k = 0

    while Epest >= Eppara:
        xr = xr_old - delta * xr_old * f(xr_old) / (
            f(xr_old + delta * xr_old) - f(xr_old)
        )

        if k > 0:
            Epest = abs((xr - xr_old) / xr) * 100

        xr_old = xr
        k += 1

    return xr, k, Epest


def imprimir_tabela(df, titulo):
    df_fmt = df.copy()

    for col in df_fmt.columns:
        if col == "Raiz":
            df_fmt[col] = df_fmt[col].map(lambda x: f"{int(x):>4d}")
        elif col == "Iterações":
            df_fmt[col] = df_fmt[col].map(lambda x: f"{int(x):>10d}")
        elif col == "Epest(%)":
            df_fmt[col] = df_fmt[col].map(lambda x: f"{x:>14.6e}")
        else:
            df_fmt[col] = df_fmt[col].map(lambda x: f"{x:>14.10f}")

    print(f"\n{titulo}")
    print(df_fmt.to_string(index=False))


# =============================================================================== #

# =============================================================================== #
# MÉTODO GRÁFICO

n = 1000
x = np.linspace(0.05, 6.5, n)

plt.figure()
plt.plot(x, f(x), "-b", label="f(x)")
plt.axhline(0, linestyle="--", color="k")
plt.scatter(2 * np.pi, 0, color="k", zorder=3)
plt.text(2 * np.pi, 0, r" $2\pi$", va="bottom")
plt.xticks(np.arange(0, 6.5 + 0.5, 0.5))
plt.xlim(-0.5, 6.75)
plt.ylim(-2, 3)
plt.xlabel("x")
plt.ylabel("f(x)")
plt.grid()
plt.legend()

# =============================================================================== #

# =============================================================================== #
# BUSCA INCREMENTAL

intervalos = [
    [2.5, 3.0],
    [3.0, 3.5],
    [4.0, 4.5],
    [4.5, 5.0],
    [5.0, 5.5],
    [6.0, 2 * np.pi],
]

xb = []

for i in range(len(intervalos)):
    xl_global = intervalos[i][0]
    xu_global = intervalos[i][1]

    x_local = np.linspace(xl_global, xu_global, n)

    for k in range(n - 1):
        xl = x_local[k]
        xu = x_local[k + 1]

        if (f(xl) * f(xu)) < 0:
            xb.append([xl, xu])

print("Subintervalos encontrados")
for i in range(len(xb)):
    print(f"S{i+1} = [{xb[i][0]:.10f}, {xb[i][1]:.10f}]")

print("\nVerificação da mudança de sinal")
for i in range(len(xb)):
    xl = xb[i][0]
    xu = xb[i][1]
    prod = f(xl) * f(xu)

    print(f"S{i+1}: f(xl)*f(xu) = {prod:.6e}")

# =============================================================================== #

# =============================================================================== #
# PARÂMETROS

Eppara = 0.5 * 10 ** (2 - 6)
delta = 0.01

# =============================================================================== #

# =============================================================================== #
# BISSECÇÃO

dados_bisseccao = []

for i in range(len(xb)):
    xl = xb[i][0]
    xu = xb[i][1]

    xr, k, Epest = bisseccao(xl, xu, Eppara)

    dados_bisseccao.append(
        {
            "Raiz": i + 1,
            "xl": xl,
            "xu": xu,
            "xr": xr,
            "Iterações": k,
            "Epest(%)": Epest,
        }
    )

df_bisseccao = pd.DataFrame(dados_bisseccao)

# =============================================================================== #

# =============================================================================== #
# FALSA POSIÇÃO

dados_falsa_posicao = []

for i in range(len(xb)):
    xl = xb[i][0]
    xu = xb[i][1]

    xr, k, Epest = falsa_posicao(xl, xu, Eppara)

    dados_falsa_posicao.append(
        {
            "Raiz": i + 1,
            "xl": xl,
            "xu": xu,
            "xr": xr,
            "Iterações": k,
            "Epest(%)": Epest,
        }
    )

df_falsa_posicao = pd.DataFrame(dados_falsa_posicao)

# =============================================================================== #

# =============================================================================== #
# NEWTON-RAPHSON

dados_newton = []

for i in range(len(xb)):
    xl = xb[i][0]
    xu = xb[i][1]
    x0 = (xl + xu) / 2

    xr, k, Epest = newton_raphson(x0, Eppara)

    dados_newton.append(
        {
            "Raiz": i + 1,
            "x0": x0,
            "xr": xr,
            "Iterações": k,
            "Epest(%)": Epest,
        }
    )

df_newton = pd.DataFrame(dados_newton)

# =============================================================================== #

# =============================================================================== #
# SECANTE

dados_secante = []

for i in range(len(xb)):
    x_im1 = xb[i][0]
    x_i = xb[i][1]

    xr, k, Epest = secante(x_im1, x_i, Eppara)

    dados_secante.append(
        {
            "Raiz": i + 1,
            "x(i-1)": x_im1,
            "x(i)": x_i,
            "xr": xr,
            "Iterações": k,
            "Epest(%)": Epest,
        }
    )

df_secante = pd.DataFrame(dados_secante)

# =============================================================================== #

# =============================================================================== #
# SECANTE MODIFICADA

dados_secante_modificada = []

for i in range(len(xb)):
    xl = xb[i][0]
    xu = xb[i][1]
    x0 = (xl + xu) / 2

    xr, k, Epest = secante_modificada(x0, Eppara, delta)

    dados_secante_modificada.append(
        {
            "Raiz": i + 1,
            "x0": x0,
            "xr": xr,
            "Iterações": k,
            "Epest(%)": Epest,
        }
    )

df_secante_modificada = pd.DataFrame(dados_secante_modificada)

# =============================================================================== #

# =============================================================================== #
# TABELAS

imprimir_tabela(df_bisseccao, "TABELA BISSECÇÃO")
imprimir_tabela(df_falsa_posicao, "TABELA FALSA POSIÇÃO")
imprimir_tabela(df_newton, "TABELA NEWTON-RAPHSON")
imprimir_tabela(df_secante, "TABELA SECANTE")
imprimir_tabela(df_secante_modificada, "TABELA SECANTE MODIFICADA")

# =============================================================================== #

# =============================================================================== #
# MOSTRAR PLOT

plt.tight_layout()
plt.show()

# =============================================================================== #
