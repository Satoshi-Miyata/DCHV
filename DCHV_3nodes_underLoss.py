import numpy as np
from scipy.optimize import root
# ========================================================================================
#Vを求める際に最適性の1次の必要条件を用いて\parL/\parV=0を解いた．
#この式はVについての線形方程式AV=0となる．
#このAについて行列式が0とならないことから，この方程式には非自明な解が存在しないことが判明した．
#つまりV=0以外の値を取り得ないので，この方法では送電を表現することが出来ない．
# ========================================================================================

# =====================================================
# 定数
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

P_lower = -10.0
P_upper = 10.0

gamma = 0.01
eps = 0.01

max_iter = 1000

# =====================================================
# qd, qs
# =====================================================

def compute_qd(p):
    return (p - a) / b


def compute_qs(p):
    return (p - c) / d


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
                (V[i]**2 - V[i]*V[j]) / R[i, j]
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
                - (V[i]**2 - V[i]*V[j]) / R[i, j]
            )

    return g


# =====================================================
# dL/dV = 0
# =====================================================

def stationarity_equation(V, p, pi_minus, pi_plus):

    F = np.zeros(n)

    for i in range(n):

        term1 = 0.0
        for m in range(n):

            if m == i:
                continue

            term1 += p[m] * V[m] / R[m, i]

        term2 = 0.0
        for l in range(n):

            if l == i:
                continue

            term2 += (2.0 * V[i] - V[l]) / R[i, l]

        term2 *= p[i]

        term3 = 0.0
        for j in range(n):

            if j == i:
                continue

            term3 += (
                (pi_minus[i, j] - pi_plus[i, j])
                * (2.0 * V[i] - V[j])
                / R[i, j]
            )

        term4 = 0.0
        for k in range(n):

            if k == i:
                continue

            term4 += (
                (pi_minus[k, i] - pi_plus[k, i])
                * V[k]
                / R[k, i]
            )

        F[i] = term1 - term2 + term3 - term4

    return F


# =====================================================
# 初期値
# =====================================================

p = np.ones(n)

pi_minus = np.ones((n, n))
pi_plus = np.ones((n, n))

for i in range(n):
    pi_minus[i, i] = 0.0
    pi_plus[i, i] = 0.0

V_guess = np.ones(n)

# =====================================================
# メインループ
# =====================================================

for itr in range(max_iter):

    p_old = p.copy()
    pi_minus_old = pi_minus.copy()
    pi_plus_old = pi_plus.copy()

    # ---------------------------------
    # qd, qs
    # ---------------------------------

    qd = compute_qd(p)
    qs = compute_qs(p)

    # ---------------------------------
    # dL/dV = 0 を解く
    # ---------------------------------

    sol = root(
        stationarity_equation,
        V_guess,
        args=(p, pi_minus, pi_plus)
    )

    if not sol.success:
        print("root solver failed")
        print(sol.message)
        break

    V = sol.x
    V_guess = V.copy()

    # ---------------------------------
    # 制約評価
    # ---------------------------------

    h = compute_h(qd, qs, V)

    g_minus = compute_g_minus(V)
    g_plus = compute_g_plus(V)

    # ---------------------------------
    # Vとhの値確認
    # ---------------------------------
    print(V)
    print(h)

    # ---------------------------------
    # 双対変数更新
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

    # 対角成分は未使用

    for i in range(n):
        pi_minus[i, i] = 0.0
        pi_plus[i, i] = 0.0

    # ---------------------------------
    # 収束判定
    # ---------------------------------

    err_p = np.max(np.abs(p - p_old))
    err_m = np.max(np.abs(pi_minus - pi_minus_old))
    err_pi = np.max(np.abs(pi_plus - pi_plus_old))

    if (
        err_p < eps
        and err_m < eps
        and err_pi < eps
    ):
        print(f"Converged at iteration {itr}")
        break

# =====================================================
# 結果
# =====================================================

print("\n===== Solution =====")

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
print(compute_h(compute_qd(p), compute_qs(p), V))

print("\ng_minus*")
print(compute_g_minus(V))

print("\ng_plus*")
print(compute_g_plus(V))