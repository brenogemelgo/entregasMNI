# ====================================================================================
#  Raízes de Funções - Questão 2
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
    return np.sin(x) + np.cos(1 + x**2) - 1


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


def imprimir_tabela(df):
    df_fmt = df.copy()

    df_fmt["x(i-1)"] = df_fmt["x(i-1)"].map(lambda x: f"{x:.10f}")
    df_fmt["x(i)"] = df_fmt["x(i)"].map(lambda x: f"{x:.10f}")
    df_fmt["xr"] = df_fmt["xr"].map(lambda x: f"{x:.10f}")
    df_fmt["Iterações"] = df_fmt["Iterações"].map(lambda x: f"{x:d}")
    df_fmt["Epest(%)"] = df_fmt["Epest(%)"].map(lambda x: f"{x:.6e}")

    print("\nTABELA SECANTE")
    print(df_fmt.to_string(index=False))


# =============================================================================== #

# =============================================================================== #
# MÉTODO GRÁFICO

n = 1000
x = np.linspace(0, 3.2, n)

plt.figure()
plt.plot(x, f(x), "-b", label="f(x)")
plt.axhline(0, linestyle="--")
plt.xlabel("x")
plt.ylabel("f(x)")
plt.legend()
plt.grid()

# =============================================================================== #

# =============================================================================== #
# SECANTE

Eppara = 0.5 * 10 ** (2 - 6)

casos = [
    ["a", 1.0, 3.0],
    ["b", 1.5, 2.5],
    ["c", 1.5, 2.25],
]

dados_secante = []

for i in range(len(casos)):
    caso = casos[i][0]
    x_im1 = casos[i][1]
    x_i = casos[i][2]

    xr, k, Epest = secante(x_im1, x_i, Eppara)

    print(f"\nCaso {caso}")
    print(f"x(i-1) = {x_im1}")
    print(f"x(i)   = {x_i}")
    print(f"Raiz encontrada = {xr:.10f}")
    print(f"Iterações = {k}")
    print(f"Epest (%) = {Epest:.6e}")

    dados_secante.append(
        {
            "Caso": caso,
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
# TABELA

imprimir_tabela(df_secante)

# =============================================================================== #

plt.show()
