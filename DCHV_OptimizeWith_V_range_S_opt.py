import numpy as np
from scipy.optimize import minimize
import sys

# =====================================================
# 問題設定
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

# 潮流上限
S_MAX = 1.0
S_MIN = -1.0

# 基準ノード電圧範囲
V1_MIN = 0.95
V1_MAX = 1.05

# 非対角要素
flow_pairs = [
    (0, 1),
    (0, 2),
    (1, 0),
    (1, 2),
    (2, 0),
    (2, 1)
]

m = len(flow_pairs)

# =====================================================
# 変数展開
# x =
# [qd(3), qs(3), V(3), S(6)]
# =====================================================

def unpack(x):

    qd = x[0:n]

    qs = x[n:2*n]

    V = x[2*n:3*n]

    Svec = x[3*n:]

    S = np.zeros((n, n))

    for k, (i, j) in enumerate(flow_pairs):
        S[i, j] = Svec[k]

    return qd, qs, V, S

# =====================================================
# 効用関数
# =====================================================

def benefit(qd):

    return np.sum(
        a * qd +
        0.5 * b * qd**2
    )

# =====================================================
# 費用関数
# =====================================================

def cost(qs):

    return np.sum(
        c * qs +
        0.5 * d * qs**2
    )

# =====================================================
# 社会厚生
# =====================================================

def welfare(qd, qs):

    return benefit(qd) - cost(qs)

# =====================================================
# 目的関数
# maximize → minimize
# =====================================================

def objective(x):

    qd, qs, _, _ = unpack(x)

    return -welfare(qd, qs)

# =====================================================
# ノード収支
# qs - qd - ΣSij = 0
# =====================================================

def create_balance_constraint(i):

    def g(x):

        qd, qs, _, S = unpack(x)

        return (
            qs[i]
            - qd[i]
            - np.sum(S[i, :])
        )

    return g

# =====================================================
# 潮流方程式
# Sij = Vi(Vi-Vj)/Rij
# =====================================================

def create_flow_constraint(i, j):

    def g(x):

        _, _, V, S = unpack(x)

        return (
            S[i, j]
            - V[i] * (V[i] - V[j]) / R[i, j]
        )

    return g

# =====================================================
# 制約
# =====================================================

constraints = []

# ノード収支制約

for i in range(n):

    constraints.append({
        'type': 'eq',
        'fun': create_balance_constraint(i)
    })

# 潮流方程式制約

for i, j in flow_pairs:

    constraints.append({
        'type': 'eq',
        'fun': create_flow_constraint(i, j)
    })

# =====================================================
# bounds
# =====================================================

bounds = []

# qd >= 0

for _ in range(n):
    bounds.append((0.0, None))

# qs >= 0

for _ in range(n):
    bounds.append((0.0, None))

# V1

bounds.append((V1_MIN, V1_MAX))

# V2, V3

bounds.append((None, None))
bounds.append((None, None))

# Sij

for _ in range(m):
    bounds.append((S_MIN, S_MAX))

# =====================================================
# 初期値
# =====================================================

x0 = np.array([

    # qd
    50.0, 50.0, 50.0,

    # qs
    50.0, 50.0, 50.0,

    # V
    1.00, 1.00, 1.00,

    # S
    0.0, 0.0,
    0.0, 0.0,
    0.0, 0.0
])

x0[2*n:3*n] = np.random.rand(n)

# =====================================================
# 求解
# =====================================================

result = minimize(
    objective,
    x0,
    method="SLSQP",
    bounds=bounds,
    constraints=constraints,
    options={
        "disp": True,
        "maxiter": 1000,
        "ftol": 1e-9
    }
)

# =====================================================
# 結果整理
# =====================================================

qd_opt, qs_opt, V_opt, S_opt = unpack(result.x)

SW_opt = welfare(qd_opt, qs_opt)

# =====================================================
# 出力
# =====================================================
print("Initial_V :", x0[2*n:])

if result.success == False:
    print("Optimization Failed")
    sys.exit()


print("S as optimization variable")


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