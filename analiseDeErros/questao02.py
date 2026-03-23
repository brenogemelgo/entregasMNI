# ====================================================================================
#  Análise de Erros - Questão 2
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
import pandas as pd
import matplotlib.pyplot as plt


# termo geral da série
def termoSerie(k):
    return 1 / (k**4)


# função principal
def estimarSerie(n):

    u = (math.pi**4) / 90
    Eppara = 0.5 * 10 ** (2 - n)

    soma = 0.0
    old = 0.0

    Ept = 100
    Epest = 100
    i = 1

    registros = []

    while Ept > Eppara:

        soma += termoSerie(i)

        Ept = abs((u - soma) / u) * 100
        if soma != 0:
            Epest = abs((soma - old) / soma) * 100
        else:
            Epest = 100

        old = soma

        registros.append(
            {
                "k": i,
                "Estimativa": soma,
                "Ept(%)": Ept,
                "Epest(%)": Epest,
            }
        )

        i += 1

    df = pd.DataFrame(registros)

    return df, u


# ============================================== TABELA ============================================== #


def tabelaConvergencia(df):

    print("\nTabela de convergência")

    print(
        df.to_string(
            index=False,
            formatters={
                "k": "{:d}".format,
                "Estimativa": "{:.10f}".format,
                "Ept(%)": "{:.6e}".format,
                "Epest(%)": "{:.6e}".format,
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

n = 6

# garante que a pasta existe
base_dir = os.path.dirname(os.path.abspath(__file__))
pasta_output = os.path.join(base_dir, "output", "series")
os.makedirs(pasta_output, exist_ok=True)

t0 = time.perf_counter()
df, u = estimarSerie(n)
t1 = time.perf_counter()
tempo_exp = t1 - t0

# tabela
tabelaConvergencia(df)

# dumps para plotar no tikz
exportarEstimativa(df, pasta_output, "estimativa_q2.dat")
exportarErros(df, pasta_output, "erros_q2.dat")

# colunas para plot
contador = df["k"]
estimativa = df["Estimativa"]
EPT = df["Ept(%)"]
EPEST = df["Epest(%)"]

# resultado final
print("\nResultado final")
print(f"Valor exato        : {u:.10f}")
print(f"Última estimativa  : {estimativa.iloc[-1]:.10f}")
print(f"Número de termos   : {int(contador.iloc[-1])}")

# tempo
print("\nTempo de execução")
print(f"t : {tempo_exp:.6e} s")

# ============================================== SUBPLOTS ============================================== #

fig, axs = plt.subplots(2, 1, figsize=(6, 8))

# estimativa
axs[0].plot(contador, estimativa, "or", label="estimativa")
axs[0].axhline(y=u, linestyle="--", label=r"$\pi^4/90$")
axs[0].set_title("Aproximação da série $\sum_{k=1}^n 1/k^4$")
axs[0].set_xlabel("Número de termos")
axs[0].set_ylabel("Estimativa")
axs[0].legend()
axs[0].grid()

# erros
axs[1].plot(contador, EPT, "og", label="$E_{\\text{pt}}$")
axs[1].plot(contador, EPEST, "ob", label="$E_{\\text{epest}}$")
axs[1].set_title("Erros da série $\sum_{k=1}^n 1/k^4$")
axs[1].set_xlabel("Número de termos")
axs[1].set_ylabel("Erro (%)")
axs[1].set_yscale("log")
axs[1].legend()
axs[1].grid()

plt.tight_layout()
plt.show()
