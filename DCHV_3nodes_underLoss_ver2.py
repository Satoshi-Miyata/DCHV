import numpy as np
from scipy.optimize import minimize

# =====================================================
#Vを求めるために，max_V Lを解いている．
#これによってq^d,q^sとの関連の中でVを決定することができる．
# =====================================================

# =====================================================
# Parameters
# =====================================================

n = 3

a = np.array([100.0, 110.0, 120.0])
b = np.array([-0.3, -0.2, -0.1])

c = np.array([5.0, 10.0, 15.0])
d = np.array([0.02, 0.04, 0.06])

R = np.array([
    [0.0, 0.2, 0.1],
    [0.2, 0.0, 0.5],
    [0.1, 0.5, 0.0]
])

P_lower = -100.0
P_upper = 100.0

gamma = 0.01
eps = 0.01

max_iter = 300

# =====================================================
# qd, qs
# =====================================================

def compute_qd(p):
    return (p - a) / b


def compute_qs(p):
    return (p - c) / d

# =====================================================
# objective function f
# =====================================================

def f_value(qd, qs):

    value = 0.0

    for i in range(n):

        value += (
            a[i] * qd[i]
            + 0.5 * b[i] * qd[i] ** 2
        )

        value -= (
            c[i] * qs[i]
            + 0.5 * d[i] * qs[i] ** 2
        )

    return value

# =====================================================
# h
# =====================================================

def compute_h(qd, qs, V):

    h = np.zeros(n)

    for i in range(n):

        flow = 0.0

        for j in range(n):

            if i == j:
                continue

            flow += (V[i] - V[j]) / R[i, j]

        h[i] = qs[i] - qd[i] - V[i] * flow

    return h

# =====================================================
# g-
# =====================================================

def compute_g_minus(V):

    g = np.zeros((n, n))

    for i in range(n):
        for j in range(n):

            if i == j:
                continue

            g[i, j] = (
                (V[i] ** 2 - V[i] * V[j]) / R[i, j]
                - P_lower
            )

    return g

# =====================================================
# g+
# =====================================================

def compute_g_plus(V):

    g = np.zeros((n, n))

    for i in range(n):
        for j in range(n):

            if i == j:
                continue

            g[i, j] = (
                P_upper
                - (V[i] ** 2 - V[i] * V[j]) / R[i, j]
            )

    return g

# =====================================================
# Lagrangian
# =====================================================

def lagrangian(V, qd, qs, p, pi_minus, pi_plus):

    h = compute_h(qd, qs, V)

    g_minus = compute_g_minus(V)

    g_plus = compute_g_plus(V)

    L = f_value(qd, qs)

    L += np.dot(p, h)

    L += np.sum(pi_minus * g_minus)

    L += np.sum(pi_plus * g_plus)

    return L

# =====================================================
# maximize L wrt V
# =====================================================

def solve_V(qd, qs, p, pi_minus, pi_plus, V0):

    def objective(V):

        return -lagrangian(
            V,
            qd,
            qs,
            p,
            pi_minus,
            pi_plus
        )

    bounds = [(0.0, None)] * n

    result = minimize(
        objective,
        V0,
        method="L-BFGS-B",
        bounds=bounds
    )

    return result.x

# =====================================================
# initialization
# =====================================================

k = 0

p = np.ones(n)

pi_minus = np.ones((n, n))
pi_plus = np.ones((n, n))

for i in range(n):
    pi_minus[i, i] = 0.0
    pi_plus[i, i] = 0.0

V_guess = np.ones(n)

# =====================================================
# main loop
# =====================================================

for k in range(max_iter):

    p_old = p.copy()
    pi_minus_old = pi_minus.copy()
    pi_plus_old = pi_plus.copy()

    # ---------------------------------
    # qd, qs
    # ---------------------------------

    qd = compute_qd(p)
    qs = compute_qs(p)

    # print(f"\nIteration {k}")
    # print(f"qd {qd}")
    # print(f"qs {qs}")

    # ---------------------------------
    # V = argmax L
    # ---------------------------------

    V = solve_V(
        qd,
        qs,
        p,
        pi_minus,
        pi_plus,
        V_guess
    )

    V_guess = V.copy()

    # ---------------------------------
    # evaluate constraints
    # ---------------------------------

    h = compute_h(qd, qs, V)

    g_minus = compute_g_minus(V)

    g_plus = compute_g_plus(V)

    # ---------------------------------
    # dual update
    # ---------------------------------

    p = p - gamma * h

    pi_minus = np.maximum(
        pi_minus - gamma * g_minus,
        0.0
    )

    pi_plus = np.maximum(
        pi_plus - gamma * g_plus,
        0.0
    )

    for i in range(n):
        pi_minus[i, i] = 0.0
        pi_plus[i, i] = 0.0

    # ---------------------------------
    # convergence check
    # ---------------------------------

    err_p = np.max(np.abs(p - p_old))
    err_m = np.max(np.abs(pi_minus - pi_minus_old))
    err_pi = np.max(np.abs(pi_plus - pi_plus_old))

    print(f"\nIteration {k}")
    print("V =", V)
    print("h =", h)

    if (
        err_p < eps
        and err_m < eps
        and err_pi < eps
    ):
        print("\nConverged")
        break

# =====================================================
# result
# =====================================================

print("\n==========================")
print("Final Result")
print("==========================")

print("p*")
print(p)

print("\nqd*")
print(compute_qd(p))

print("\nqs*")
print(compute_qs(p))

print("\nV*")
print(V)

print("\npi_minus*")
print(pi_minus)

print("\npi_plus*")
print(pi_plus)

print("\nh*")
print(compute_h(
    compute_qd(p),
    compute_qs(p),
    V
))