# ====================================================================================
#  Análise de Erros - Questão 1
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

import math
import time
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================== TERMOS: EXPLÍCITO E RECURSIVO ============================================== #


# termo geral explícito
def termoMaclaurinExplicito(x, k):
    return ((-1) ** k) * x ** (2 * k) / math.factorial(k)


# termo geral em forma relação recursiva
def termoMaclaurinGaussiana(x, k, termoAnterior):
    return termoAnterior * (-(x**2)) / k


# ============================================== MÉTODO RECURSIVO ============================================== #


def estimarGaussianaRecursiva(x, n):

    u = math.exp(-(x**2))
    Eppara = 0.5 * 10 ** (2 - n)

    soma = 1.0
    termo = 1.0
    old = soma

    Epest = 100
    i = 0

    registros = [
        {
            "k": 0,
            "Estimativa": soma,
            "Ept(%)": abs((u - soma) / u) * 100,
            "Epest(%)": np.inf,
        }
    ]

    while Epest > Eppara:

        i += 1

        termo = termoMaclaurinGaussiana(x, i, termo)
        soma += termo

        Ept = abs((u - soma) / u) * 100
        if soma != 0:
            Epest = abs((soma - old) / soma) * 100
        else:
            Epest = 100

        old = soma

        registros.append({"k": i, "Estimativa": soma, "Ept(%)": Ept, "Epest(%)": Epest})

    df = pd.DataFrame(registros)

    return df, u


# ============================================== MÉTODO EXPLÍCITO ============================================== #


def estimarGaussianaExplicita(x, n):

    Eppara = 0.5 * 10 ** (2 - n)

    soma = 0
    old = 0

    Epest = 100
    i = 0

    while Epest > Eppara:

        termo = termoMaclaurinExplicito(x, i)
        soma += termo

        if soma != 0:
            Epest = abs((soma - old) / soma) * 100
        else:
            Epest = 100

        old = soma

        i += 1

    return soma


# ============================================== TABELA ============================================== #


def tabelaConvergencia(df, u):

    print(f"\nValor real = {u:.16f}")
    print("\nTabela de convergência")

    print(
        df.copy().to_string(
            index=False,
            formatters={
                "k": "{:d}".format,
                "Estimativa": "{:.10f}".format,
                "Ept(%)": "{:.6e}".format,
                "Epest(%)": lambda v: "inf" if np.isinf(v) else f"{v:.6e}",
            },
        )
    )


# ============================================== EXPORTAÇÃO DAT ============================================== #


def exportarEstimativa(df, pasta, nome):

    caminho = os.path.join(pasta, nome)
    df[["k", "Estimativa"]].to_csv(caminho, sep=" ", index=False, header=False)


def exportarErros(df, pasta, nome):

    caminho = os.path.join(pasta, nome)
    df[["k", "Ept(%)", "Epest(%)"]].to_csv(caminho, sep=" ", index=False, header=False)


# ============================================== PROGRAMA PRINCIPAL ============================================== #

valores_x = [5, 6]

n = 6

# garante que a pasta existe
base_dir = os.path.dirname(os.path.abspath(__file__))
pasta_output = os.path.join(base_dir, "output", "series")
os.makedirs(pasta_output, exist_ok=True)

for x in valores_x:

    print("\n==============================================")
    print(f"x = {x}")
    print("==============================================")

    # recursivo
    t0 = time.perf_counter()
    df, u = estimarGaussianaRecursiva(x, n)
    t1 = time.perf_counter()
    tempo_rec = t1 - t0

    # explícito
    t0 = time.perf_counter()
    estimarGaussianaExplicita(x, n)
    t1 = time.perf_counter()
    tempo_exp = t1 - t0

    # tabela
    tabelaConvergencia(df, u)

    # dumps para plotar no tikz
    exportarEstimativa(df, pasta_output, f"estimativa_x{x}.dat")
    exportarErros(df, pasta_output, f"erros_x{x}.dat")

    # tempos
    print("\nTempos de execução")
    print(f"Explícito : {tempo_exp:.6e} s")
    print(f"Recursivo : {tempo_rec:.6e} s")

    # colunas para plot
    contador = df["k"]
    estimativa = df["Estimativa"]
    EPT = df["Ept(%)"]
    EPEST = df["Epest(%)"]

    # ============================================== SUBPLOTS ============================================== #

    fig, axs = plt.subplots(2, 1, figsize=(6, 8))

    # estimativa
    axs[0].plot(contador, estimativa, "or", label="estimativa")
    axs[0].axhline(y=u, linestyle="--", label="$\mathrm{e}^{-x^2}$")
    axs[0].set_title(f"Aproximação da Gaussiana (x = {x})")
    axs[0].set_xlabel("Número de termos")
    axs[0].set_ylabel("Estimativa")
    axs[0].legend()
    axs[0].grid()

    # erros
    axs[1].plot(contador, EPT, "og", label="$E_{\\text{pt}}$")
    axs[1].plot(contador, EPEST, "ob", label="$E_{\\text{pest}}$")
    axs[1].set_title(f"Erros para x = {x}")
    axs[1].set_xlabel("Número de termos")
    axs[1].set_ylabel("Erro (%)")
    axs[1].set_yscale("log")
    axs[1].legend()
    axs[1].grid()

    plt.tight_layout()

plt.show()
