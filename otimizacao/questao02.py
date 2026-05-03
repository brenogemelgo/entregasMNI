from scipy.optimize import minimize

# =============================================================================== #
# GLOBAL


def P(x):
    p, q = x
    return 2 * p + 2 * q - 2 * p**2 - 2 * q**2 - 2 * p * q


def f(x):
    return -P(x)


# =============================================================================== #

# =============================================================================== #
# PARÂMETROS

x0 = [0.25, 0.25]
x_analitico = [1 / 3, 1 / 3]

# =============================================================================== #

# =============================================================================== #
# OTIMIZAÇÃO MULTIDIMENSIONAL

opt = minimize(f, x0=x0)

p = opt.x[0]
q = opt.x[1]
r = 1 - p - q

P_max = -opt.fun
Ept = abs((x_analitico - opt.x) / x_analitico) * 100

print("\n" + "=" * 67)
print("OTIMIZAÇÃO MULTIDIMENSIONAL")
print("=" * 67)
print(f"p = {p:.16f}")
print(f"q = {q:.16f}")
print(f"r = {r:.16f}")
print(f"P_max = {P_max:.16f}")
print(f"Ept_p (%) = {Ept[0]:.16f}")
print(f"Ept_q (%) = {Ept[1]:.16f}")
if Ept[0] == Ept[1]:
    print("Erros de p e q são idênticos! Reportar apenas um basta.")
print("Iterações =", opt.nit)
print("Sucesso =", opt.success)
print("=" * 67)

# =============================================================================== #
