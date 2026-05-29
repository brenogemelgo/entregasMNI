import numpy as np
import matplotlib.pyplot as plt

Eppara = 0.5 * 10 ** (2 - 6)
maxit = 1000

# chutes obtidos pelo método gráfico
chutes = [
    np.array([-0.5, -0.2], dtype=float),
    np.array([-0.2, 0.3], dtype=float),
    np.array([1.2, 0.2], dtype=float),
]


def f1(x, y):
    return y + x**2 - x - 0.5


def f2(x, y):
    return y + 5 * x * y - x**2


def df1dx(x, y):
    return 2 * x - 1


def df1dy(x, y):
    return 1


def df2dx(x, y):
    return 5 * y - 2 * x


def df2dy(x, y):
    return 1 + 5 * x


def funcEpest(x_new, x_old):
    erros = np.zeros(len(x_new))

    for i in range(len(x_new)):
        if x_new[i] != 0:
            erros[i] = abs((x_new[i] - x_old[i]) / x_new[i]) * 100
        else:
            erros[i] = 100

    return np.max(erros)


def newtonRaphson(chute, Eppara, maxit):
    x_old = chute.copy()
    x_new = chute.copy()

    Epest = 100
    k = 0

    while Epest > Eppara and k < maxit:
        x = x_old[0]
        y = x_old[1]

        f1_i = f1(x, y)
        f2_i = f2(x, y)

        df1dx_i = df1dx(x, y)
        df1dy_i = df1dy(x, y)
        df2dx_i = df2dx(x, y)
        df2dy_i = df2dy(x, y)

        denominador = df1dx_i * df2dy_i - df1dy_i * df2dx_i

        x_new[0] = x_old[0] - (f1_i * df2dy_i - f2_i * df1dy_i) / denominador
        x_new[1] = x_old[1] - (f2_i * df1dx_i - f1_i * df2dx_i) / denominador

        Epest = funcEpest(x_new, x_old)

        x_old = x_new.copy()
        k += 1

    return x_new, Epest, k


def curva1(x):
    return 0.5 - x**2 + x


def curva2(x):
    return x**2 / (1 + 5 * x)


def gerarGrafico():
    x = np.linspace(-1.0, 1.6, 2000)

    y1 = curva1(x)
    y2 = curva2(x)

    y2[np.abs(1 + 5 * x) < 0.03] = np.nan  # evita assíntota

    plt.figure(figsize=(8, 5))
    plt.plot(x, y1, label=r"$y=\frac{1}{2}-x^2+x$")
    plt.plot(x, y2, label=r"$y=\frac{x^2}{1+5x}$")

    plt.axhline(0, linewidth=0.8)
    plt.axvline(0, linewidth=0.8)

    plt.xlim(-0.8, 1.4)
    plt.ylim(-0.4, 0.9)

    plt.xticks(np.arange(-0.8, 1.41, 0.1))
    plt.yticks(np.arange(-0.4, 0.91, 0.1))

    plt.xlabel("x")
    plt.ylabel("y")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()


def imprimirLinha(nome, chute, sol, Epest, k):
    print(
        f"{nome:10s} "
        f"{chute[0]:12.6f} {chute[1]:12.6f} "
        f"{sol[0]:12.6f} {sol[1]:12.6f} "
        f"{Epest:14.10f} {k:10d}"
    )


gerarGrafico()

print("Sistema não linear")
print(f"Eppara = {Eppara:.8f}%")
print(f"maxit = {maxit}")
print()
print(
    f"{'Solução':10s} "
    f"{'x0':>12s} {'y0':>12s} "
    f"{'x':>12s} {'y':>12s} "
    f"{'Epest (%)':>14s} {'Iterações':>10s}"
)
print("-" * 90)

for i in range(len(chutes)):
    sol, Epest, k = newtonRaphson(chutes[i], Eppara, maxit)
    imprimirLinha(f"S{i + 1}", chutes[i], sol, Epest, k)

plt.show()
