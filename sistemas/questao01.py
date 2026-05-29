import numpy as np

A = np.array([[3, -1, 0], [-1, 4, -3], [0, -3, 5]], dtype=float)
b = np.array([60, 0, 20], dtype=float)

Eppara = 0.5 * 10 ** (2 - 6)
maxit = 1000
lambida = 1.5


def funcEpest(x_new, x_old):
    erros = np.zeros(len(x_new))

    for i in range(len(x_new)):
        if x_new[i] != 0:
            erros[i] = abs((x_new[i] - x_old[i]) / x_new[i]) * 100
        else:
            erros[i] = 100

    return np.max(erros)


def gaussPivotamento(A, b):
    Aum = np.hstack((A.copy(), b.reshape(-1, 1)))
    n = len(b)

    # eliminação progressiva com pivotamento parcial
    for i in range(n - 1):
        pivo = i + np.argmax(np.abs(Aum[i:n, i]))

        if pivo != i:
            Aum[[i, pivo], :] = Aum[[pivo, i], :]

        for j in range(i + 1, n):
            fator = Aum[j, i] / Aum[i, i]
            Aum[j, i : n + 1] = Aum[j, i : n + 1] - fator * Aum[i, i : n + 1]

    # substituição regressiva
    x = np.zeros(n)
    x[n - 1] = Aum[n - 1, n] / Aum[n - 1, n - 1]

    for i in range(n - 2, -1, -1):
        soma = 0

        for j in range(i + 1, n):
            soma += Aum[i, j] * x[j]

        x[i] = (Aum[i, n] - soma) / Aum[i, i]

    return x


def jacobi(A, b, Eppara, maxit):
    n = len(b)
    x_old = np.zeros(n)
    x_new = np.zeros(n)

    Epest = 100
    k = 0

    while Epest > Eppara and k < maxit:
        for i in range(n):
            soma = 0

            for j in range(n):
                if j != i:
                    soma += A[i, j] * x_old[j]

            x_new[i] = (b[i] - soma) / A[i, i]

        Epest = funcEpest(x_new, x_old)
        x_old = x_new.copy()
        k += 1

    return x_new, Epest, k


def gaussSeidel(A, b, Eppara, maxit):
    n = len(b)
    x_old = np.zeros(n)
    x_new = np.zeros(n)

    Epest = 100
    k = 0

    while Epest > Eppara and k < maxit:
        for i in range(n):
            soma_A = 0
            soma_B = 0

            for j in range(n):
                if j < i:
                    soma_A += A[i, j] * x_new[j]
                elif j > i:
                    soma_B += A[i, j] * x_old[j]

            x_new[i] = (b[i] - soma_A - soma_B) / A[i, i]

        Epest = funcEpest(x_new, x_old)
        x_old = x_new.copy()
        k += 1

    return x_new, Epest, k


def gaussSeidelRelaxamento(A, b, Eppara, maxit, lambida):
    n = len(b)
    x_old = np.zeros(n)
    x_new = np.zeros(n)

    Epest = 100
    k = 0

    while Epest > Eppara and k < maxit:
        for i in range(n):
            soma_A = 0
            soma_B = 0

            for j in range(n):
                if j < i:
                    soma_A += A[i, j] * x_new[j]
                elif j > i:
                    soma_B += A[i, j] * x_old[j]

            valor = (b[i] - soma_A - soma_B) / A[i, i]
            x_new[i] = lambida * valor + (1 - lambida) * x_old[i]

        Epest = funcEpest(x_new, x_old)
        x_old = x_new.copy()
        k += 1

    return x_new, Epest, k


def imprimir_linha(nome, x, Epest=None, k=None):
    if Epest is None:
        print(
            f"{nome:32s} {x[0]:12.6f} {x[1]:12.6f} {x[2]:12.6f} {'--':>14s} {'--':>10s}"
        )
    else:
        print(
            f"{nome:32s} {x[0]:12.6f} {x[1]:12.6f} {x[2]:12.6f} {Epest:14.8f} {k:10d}"
        )


x_gauss = gaussPivotamento(A, b)
x_jacobi, Epest_jacobi, k_jacobi = jacobi(A, b, Eppara, maxit)
x_gs, Epest_gs, k_gs = gaussSeidel(A, b, Eppara, maxit)
x_relax, Epest_relax, k_relax = gaussSeidelRelaxamento(A, b, Eppara, maxit, lambida)

print("Resultados para o sistema da Lei de Darcy")
print(f"Eppara = {Eppara:.8f}%")
print(f"lambda = {lambida}")
print()
print(
    f"{'Método':32s} {'p1 (MPa)':>12s} {'p2 (MPa)':>12s} {'p3 (MPa)':>12s} {'Epest (%)':>14s} {'Iterações':>10s}"
)
print("-" * 98)

imprimir_linha("Gauss com pivotamento", x_gauss)
imprimir_linha("Jacobi", x_jacobi, Epest_jacobi, k_jacobi)
imprimir_linha("Gauss-Seidel", x_gs, Epest_gs, k_gs)
imprimir_linha("Gauss-Seidel com relaxamento", x_relax, Epest_relax, k_relax)
