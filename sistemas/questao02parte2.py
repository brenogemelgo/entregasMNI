import numpy as np

sistemas = [
    {
        "nome": "Conjunto 1",
        "A": np.array([[8, 3, 1], [-6, 0, 7], [2, 4, -1]], dtype=float),
        "b": np.array([12, 1, 5], dtype=float),
        "ordem": [0, 2, 1],  # (1, 3, 2)
    },
    {
        "nome": "Conjunto 2",
        "A": np.array([[1, 1, 5], [-1, 4, -1], [3, 1, -1]], dtype=float),
        "b": np.array([7, 4, 3], dtype=float),
        "ordem": [2, 1, 0],  # (3, 2, 1)
    },
    {
        "nome": "Conjunto 3",
        "A": np.array([[-7, 3, 5], [-2, 4, -5], [0, 2, -1]], dtype=float),
        "b": np.array([7, -3, 1], dtype=float),
        "ordem": [0, 2, 1],  # (1, 3, 2)
    },
]

maxit = 1000
Eppara = 0.5 * 10 ** (2 - 6)


def funcEpest(x_new, x_old):
    erros = np.zeros(len(x_new))

    for i in range(len(x_new)):
        if x_new[i] != 0:
            erros[i] = abs((x_new[i] - x_old[i]) / x_new[i]) * 100
        else:
            erros[i] = 100

    return np.max(erros)


def reordenarSistema(A, b, ordem):
    A_new = A[ordem, :].copy()
    b_new = b[ordem].copy()

    return A_new, b_new


def jacobi(A, b, maxit):
    n = len(b)
    x_old = np.zeros(n)
    x_new = np.zeros(n)

    Epest = 100

    for k in range(1, maxit + 1):
        for i in range(n):
            soma = 0

            for j in range(n):
                if j != i:
                    soma += A[i, j] * x_old[j]

            x_new[i] = (b[i] - soma) / A[i, i]

        Epest = funcEpest(x_new, x_old)

        if not np.all(np.isfinite(x_new)):
            return x_new, Epest, k, "divergiu"

        if Epest <= Eppara:
            return x_new, Epest, k, "convergiu"

        x_old = x_new.copy()

    return x_new, Epest, maxit, "não convergiu"


def gaussSeidel(A, b, maxit):
    n = len(b)
    x_old = np.zeros(n)
    x_new = np.zeros(n)

    Epest = 100

    for k in range(1, maxit + 1):
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

        if not np.all(np.isfinite(x_new)):
            return x_new, Epest, k, "divergiu"

        if Epest <= Eppara:
            return x_new, Epest, k, "convergiu"

        x_old = x_new.copy()

    return x_new, Epest, maxit, "não convergiu"


def imprimirDiagonal(A):
    diag = [A[i, i] for i in range(len(A))]
    return f"({diag[0]:.0f}, {diag[1]:.0f}, {diag[2]:.0f})"


def imprimirLinha(conjunto, metodo, x, Epest, k, status):
    print(
        f"{conjunto:12s} {metodo:14s} "
        f"{x[0]:14.6e} {x[1]:14.6e} {x[2]:14.6e} "
        f"{Epest:14.6e} {k:10d} {status:>15s}"
    )


print("Aplicação dos métodos iterativos com reordenação")
print(f"Eppara = {Eppara:.8f}%")
print(f"maxit = {maxit}")
print()
print(
    f"{'Conjunto':12s} {'Método':14s} "
    f"{'x':>14s} {'y':>14s} {'z':>14s} "
    f"{'Epest (%)':>14s} {'Iterações':>10s} {'Status':>15s}"
)
print("-" * 115)

for sistema in sistemas:
    nome = sistema["nome"]
    A = sistema["A"]
    b = sistema["b"]
    ordem = sistema["ordem"]

    A_ord, b_ord = reordenarSistema(A, b, ordem)

    print()
    print(f"{nome}: ordem aplicada = {ordem}, diagonal = {imprimirDiagonal(A_ord)}")

    x_jacobi, Epest_jacobi, k_jacobi, status_jacobi = jacobi(A_ord, b_ord, maxit)
    x_gs, Epest_gs, k_gs, status_gs = gaussSeidel(A_ord, b_ord, maxit)

    imprimirLinha(nome, "Jacobi", x_jacobi, Epest_jacobi, k_jacobi, status_jacobi)
    imprimirLinha(nome, "Gauss-Seidel", x_gs, Epest_gs, k_gs, status_gs)
