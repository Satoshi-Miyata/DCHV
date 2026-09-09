import numpy as np
from scipy.optimize import minimize
import sys

#V_1を基準ノードとして上限値と下限値の制約を設けた．これによって基準電圧を直接定めるよりは自由度のある最適化ができるはず．

# ==========================
# パラメータ
# ==========================

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

Vmin = 0.95
Vmax = 1.05

Smin = -1.0
Smax = 1.0

# ==========================
# 添字取り出し
# ==========================

def unpack(x):
    qd = x[0:n]
    qs = x[n:2*n]
    V  = x[2*n:3*n]
    return qd, qs, V

# ==========================
# 潮流計算
# ==========================

def compute_S(V):

    S = np.zeros((n, n))

    for i in range(n):
        for j in range(n):

            if i == j:
                continue

            S[i, j] = V[i] * (V[i] - V[j]) / R[i, j]

    return S

# ==========================
# 社会厚生
# ==========================

def welfare(qd, qs):

    benefit = np.sum(
        a * qd +
        0.5 * b * qd**2
    )

    cost = np.sum(
        c * qs +
        0.5 * d * qs**2
    )

    return benefit - cost

# ==========================
# 目的関数
# scipyは最小化なので符号反転
# ==========================

def objective(x):

    qd, qs, V = unpack(x)

    return -welfare(qd, qs)

# ==========================
# 電力収支制約
# qs - qd - ΣSij = 0
# ==========================

def power_balance(i):

    def constraint(x):

        qd, qs, V = unpack(x)

        S = compute_S(V)

        return qs[i] - qd[i] - np.sum(S[i, :])

    return constraint

# ==========================
# 潮流上下限制約
# Smax - Sij >= 0
# Sij - Smin >= 0
# ==========================

ineq_constraints = []

for i in range(n):
    for j in range(n):

        if i == j:
            continue

        def upper(x, i=i, j=j):

            _, _, V = unpack(x)

            S = compute_S(V)

            return Smax - S[i, j]

        def lower(x, i=i, j=j):

            _, _, V = unpack(x)

            S = compute_S(V)

            return S[i, j] - Smin

        ineq_constraints.append({
            'type': 'ineq',
            'fun': upper
        })

        ineq_constraints.append({
            'type': 'ineq',
            'fun': lower
        })

# ==========================
# 等式制約
# ==========================

eq_constraints = []

for i in range(n):
    eq_constraints.append({
        'type': 'eq',
        'fun': power_balance(i)
    })

constraints = eq_constraints + ineq_constraints

# ==========================
# 変数境界
# ==========================

bounds = []

# qd >= 0
for _ in range(n):
    bounds.append((0, None))

# qs >= 0
for _ in range(n):
    bounds.append((0, None))

# V1
bounds.append((Vmin, Vmax))

# V2,V3
bounds.append((None, None))
bounds.append((None, None))

# ==========================
# 初期値
# ==========================

x0 = np.array([
    50, 50, 50,
    50, 50, 50,
    1.00, 1.00, 1.00
])

x0[2*n:] = np.random.rand(n)



# ==========================
# 最適化実行
# ==========================

result = minimize(
    objective,
    x0,
    method='SLSQP',
    bounds=bounds,
    constraints=constraints,
    options={
        'maxiter': 1000,
        'ftol': 1e-9,
        'disp': True
    }
)

# ==========================
# 結果表示
# ==========================
print("Initial_V :", x0[2*n:])

if result.success == False:
    print("Optimization Failed")
    sys.exit()

qd_opt, qs_opt, V_opt = unpack(result.x)
S_opt = compute_S(V_opt)

print("S as dependent variable")

print("\n===== Optimization Result =====")
print("Success :", result.success)
print("Message :", result.message)

print("\nqd*")
print(qd_opt)

print("\nqs*")
print(qs_opt)

print("\nV*")
print(V_opt)

print("\nS*")
print(S_opt)

print("\Loss")
print(S_opt + S_opt.T)

print("\nSocial Welfare")
print(welfare(qd_opt, qs_opt))

# 収支確認
print("\nBalance Check")

for i in range(n):

    balance = (
        qs_opt[i]
        - qd_opt[i]
        - np.sum(S_opt[i, :])
    )

    print(f"Node {i+1}: {balance:.6e}")