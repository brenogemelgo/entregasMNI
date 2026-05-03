import numpy as np
import matplotlib.pyplot as plt
import math

# =============================================================================== #
# GLOBAL


def C(v):
    return v**3 * 600 / (v - 0.35)


def buscaDaRazaoAurea(phi, vl, vu, Eppara):
    Ept = 0
    Ea = 100
    i = 0

    while Ea >= Eppara:
        d = (phi - 1) * (vu - vl)

        v1 = vl + d
        v2 = vu - d

        if C(v1) < C(v2):
            vl = v2
            v_opt = v1
        else:
            vu = v1
            v_opt = v2

        Ept = abs((0.525 - v_opt) / 0.525) * 100
        Ea = (2 - phi) * abs((vu - vl) / v_opt) * 100
        i += 1

    return v_opt, Ept, Ea, i


def interpolacaoQuadratica(v1, v2, v3, Eppara):
    Epest = 100
    v_opt_old = v2
    i = 0

    while Epest >= Eppara:

        v_opt = v2 - 0.5 * (
            (v2 - v1) ** 2 * (C(v2) - C(v3)) - (v2 - v3) ** 2 * (C(v2) - C(v1))
        ) / ((v2 - v1) * (C(v2) - C(v3)) - (v2 - v3) * (C(v2) - C(v1)))

        Ept = abs((0.525 - v_opt) / 0.525) * 100
        if i > 0:
            Epest = abs((v_opt - v_opt_old) / v_opt) * 100

        if v_opt > v2:
            if C(v_opt) < C(v2):
                v1 = v2
                v2 = v_opt
            else:
                v3 = v_opt
        else:
            if C(v_opt) < C(v2):
                v3 = v2
                v2 = v_opt
            else:
                v1 = v_opt

        v_opt_old = v_opt
        i += 1

    return v_opt, Ept, Epest, i


# =============================================================================== #

# =============================================================================== #
# PARÂMETROS

Eppara = 0.5 * 10 ** (2 - 6)
phi = (1 + math.sqrt(5)) / 2

v1 = 0.36
v2 = 0.5
v3 = 0.75

# =============================================================================== #

# =============================================================================== #
# BUSCA DA RAZÃO ÁUREA

v_opt, Ept, Ea, i = buscaDaRazaoAurea(phi, v1, v3, Eppara)

print("\n" + "=" * 67)
print("BUSCA DA RAZÃO ÁUREA")
print("=" * 67)
print(f"v_opt = {v_opt:.16f}")
print(f"C(v_opt) = {C(v_opt):.16f}")
print(f"Ept (%) = {Ept:.16f}")
print(f"Ea (%) = {Ea:.16f}")
print("Iterações = ", i)
print("=" * 67)

# =============================================================================== #

# =============================================================================== #
# INTERPOLAÇÃO QUADRÁTICA

v_opt, Ept, Epest, i = interpolacaoQuadratica(v1, v2, v3, Eppara)

print("\n" + "=" * 67)
print("INTERPOLAÇÃO QUADRÁTICA")
print("=" * 67)
print(f"v_opt = {v_opt:.16f}")
print(f"C(v_opt) = {C(v_opt):.16f}")
print(f"Ept (%) = {Ept:.16f}")
print(f"Epest (%) = {Epest:.16f}")
print("Iterações = ", i)
print("=" * 67)

# =============================================================================== #
