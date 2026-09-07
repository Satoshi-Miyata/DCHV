import numpy as np
from scipy.optimize import minimize

# ==================================================
# パラメータ
# ==================================================

n = 3

a = np.array([100, 110, 120])
b = np.array([-0.3, -0.2, -0.1])

c = np.array([5, 10, 15])
d = np.array([0.02, 0.04, 0.06])

R = np.array([
    [0.0, 0.2, 0.1],
    [0.2, 0.0, 0.5],
    [0.1, 0.5, 0.0]
])

# ==================================================
# Sのインデックス管理
# ==================================================

flow_pairs = [
    (0, 1),
    (0, 2),
    (1, 0),
    (1, 2),
    (2, 0),
    (2, 1)
]

m = len(flow_pairs)


def build_S(flow):

    S = np.zeros((n, n))

    for k, (i, j) in enumerate(flow_pairs):
        S[i, j] = flow[k]

    return S


# ==================================================
# 目的関数
# maximize welfare
# ↓
# minimize -welfare
# ==================================================

def objective(x):

    qd = x[:n]
    qs = x[n:2*n]

    B = a * qd + 0.5 * b * qd**2
    C = c * qs + 0.5 * d * qs**2

    welfare = np.sum(B - C)

    return -welfare


# ==================================================
# 需給バランス制約
# ==================================================

def balance_constraint(x):

    qd = x[:n]
    qs = x[n:2*n]

    flow = x[2*n:2*n+m]

    S = build_S(flow)

    h = np.zeros(n)

    for i in range(n):

        h[i] = (
            qs[i]
            - qd[i]
            - np.sum(S[i, :])
        )

    return h


# ==================================================
# 潮流方程式
# Sij=(Vi^2-ViVj)/Rij
# ==================================================

def power_flow_constraint(x):

    flow = x[2*n:2*n+m]
    V = x[2*n+m:]

    S = build_S(flow)

    residual = []

    for i, j in flow_pairs:

        residual.append(
            S[i, j]
            - (V[i]**2 - V[i]*V[j]) / R[i, j]
        )

    return np.array(residual)


# ==================================================
# 基準電圧
# V1 = 1
# ==================================================

def voltage_reference_constraint(x):

    V = x[2*n+m:]

    return np.array([
        V[0] - 1.0
    ])


# ==================================================
# 制約
# ==================================================

constraints = [

    {
        "type": "eq",
        "fun": balance_constraint
    },

    {
        "type": "eq",
        "fun": power_flow_constraint
    },

    {
        "type": "eq",
        "fun": voltage_reference_constraint
    }

]

# ==================================================
# bounds
# ==================================================

bounds = []

# qd >= 0
for _ in range(n):
    bounds.append((0, None))

# qs >= 0
for _ in range(n):
    bounds.append((0, None))

# flow
for _ in range(m):
    bounds.append((None, None))

# voltage
for _ in range(n):
    bounds.append((None, None))

# ==================================================
# 初期値
# ==================================================

x0 = np.zeros(2*n + m + n)

# qd
x0[:n] = 50

# qs
x0[n:2*n] = 50

# flow
x0[2*n:2*n+m] = 0

# voltage
x0[2*n+m:] = 1

# ==================================================
# 最適化
# ==================================================

result = minimize(
    objective,
    x0,
    method="SLSQP",
    constraints=constraints,
    bounds=bounds,
    options={
        "disp": True,
        "maxiter": 1000,
        "ftol": 1e-9
    }
)

# ==================================================
# 結果取り出し
# ==================================================

qd_opt = result.x[:n]

qs_opt = result.x[n:2*n]

flow_opt = result.x[2*n:2*n+m]

V_opt = result.x[2*n+m:]

S_opt = build_S(flow_opt)

# ==================================================
# 結果表示
# ==================================================

print("\n====================")
print("Optimization Result")
print("====================")

print("success :", result.success)
print("status  :", result.status)
print("message :", result.message)

print("\nSocial Welfare")
print(-result.fun)

print("\nqd*")
print(qd_opt)

print("\nqs*")
print(qs_opt)

print("\nV*")
print(V_opt)

print("\nS*")
print(S_opt)

# ==================================================
# 制約残差チェック
# ==================================================

print("\n====================")
print("Constraint Residual")
print("====================")

print("\nBalance constraint")
print(balance_constraint(result.x))

print("\nPower flow constraint")
print(power_flow_constraint(result.x))

print("\nReference voltage")
print(voltage_reference_constraint(result.x))

# ==================================================
# 各枝の潮流確認
# ==================================================

print("\n====================")
print("Branch Flow Check")
print("====================")

for i, j in flow_pairs:

    lhs = S_opt[i, j]

    rhs = (
        V_opt[i]**2
        - V_opt[i]*V_opt[j]
    ) / R[i, j]

    print(
        f"S{i+1}{j+1}: "
        f"{lhs:.6f}   "
        f"theory={rhs:.6f}   "
        f"error={lhs-rhs:.3e}"
    )

# import numpy as np
# from scipy.optimize import minimize

# #ロスが物理モデルに即した形で表現されるように，電圧制約を追加したネットワークモデル．

# # =====================================================
# # パラメータ
# # =====================================================

# n = 3

# a = np.array([100, 110, 120])
# b = np.array([-0.3, -0.2, -0.1])

# c = np.array([5, 10, 15])
# d = np.array([0.02, 0.04, 0.06])

# R = np.array([
#     [0.0, 0.2, 0.1],
#     [0.2, 0.0, 0.5],
#     [0.1, 0.5, 0.0]
# ])

# # =====================================================
# # 送電変数
# # =====================================================

# flow_pairs = [
#     (0, 1),
#     (0, 2),
#     (1, 0),
#     (1, 2),
#     (2, 0),
#     (2, 1)
# ]

# m = len(flow_pairs)

# # =====================================================
# # flow → S行列
# # =====================================================

# def build_S(flow):

#     S = np.zeros((n, n))

#     for k, (i, j) in enumerate(flow_pairs):
#         S[i, j] = flow[k]

#     return S


# # =====================================================
# # 目的関数
# # maximize welfare
# # ↓
# # minimize(-welfare)
# # =====================================================

# def objective(x):

#     qd = x[:n]
#     qs = x[n:2*n]

#     B = a * qd + 0.5 * b * qd**2
#     C = c * qs + 0.5 * d * qs**2

#     welfare = np.sum(B - C)

#     return -welfare


# # =====================================================
# # 需給バランス制約
# # qs_i - qd_i - ΣSij = 0
# # =====================================================

# def balance_constraint(x):

#     qd = x[:n]
#     qs = x[n:2*n]

#     flow = x[2*n : 2*n+m]

#     S = build_S(flow)

#     h = np.zeros(n)

#     for i in range(n):
#         h[i] = qs[i] - qd[i] - np.sum(S[i, :])

#     return h


# # =====================================================
# # 電圧制約
# # (Vi - Vj)^2 = Rij(Sij + Sji)
# # =====================================================

# def voltage_constraint(x):

#     flow = x[2*n : 2*n+m]
#     V = x[2*n+m:]

#     S = build_S(flow)

#     g = []

#     for i in range(n):
#         for j in range(i + 1, n):

#             g.append(
#                 (V[i] - V[j])**2
#                 - R[i, j] * (S[i, j] + S[j, i])
#             )

#     return np.array(g)


# # =====================================================
# # 基準電圧
# # V1 = 1
# # =====================================================

# def reference_voltage(x):

#     V = x[2*n+m:]

#     return np.array([V[0] - 1.0])


# # =====================================================
# # 制約
# # =====================================================

# constraints = [

#     {
#         "type": "eq",
#         "fun": balance_constraint
#     },

#     {
#         "type": "eq",
#         "fun": voltage_constraint
#     },

#     {
#         "type": "eq",
#         "fun": reference_voltage
#     }
# ]

# # =====================================================
# # Bounds
# # =====================================================

# bounds = []

# # qd >= 0
# for _ in range(n):
#     bounds.append((0, None))

# # qs >= 0
# for _ in range(n):
#     bounds.append((0, None))

# # S は自由
# for _ in range(m):
#     bounds.append((None, None))

# # V は自由
# for _ in range(n):
#     bounds.append((None, None))


# # =====================================================
# # 初期値
# # =====================================================

# x0 = np.zeros(2*n + m + n)

# x0[:n] = 50.0
# x0[n:2*n] = 50.0

# x0[2*n : 2*n+m] = 0.0

# # V=[1,1,1]
# x0[2*n+m:] = 1.0


# # =====================================================
# # 最適化
# # =====================================================

# result = minimize(
#     objective,
#     x0,
#     method="SLSQP",
#     bounds=bounds,
#     constraints=constraints,
#     options={
#         "maxiter": 1000,
#         "ftol": 1e-9,
#         "disp": True
#     }
# )

# # =====================================================
# # 結果取り出し
# # =====================================================

# qd_opt = result.x[:n]

# qs_opt = result.x[n:2*n]

# flow_opt = result.x[2*n : 2*n+m]

# V_opt = result.x[2*n+m:]

# S_opt = build_S(flow_opt)

# # =====================================================
# # 結果表示
# # =====================================================

# print("\n==============================")
# print("Optimization Result")
# print("==============================")

# print("success =", result.success)
# print("status  =", result.status)
# print("message =", result.message)

# print("\nObjective Value")
# print(-result.fun)

# print("\nqd*")
# print(qd_opt)

# print("\nqs*")
# print(qs_opt)

# print("\nV*")
# print(V_opt)

# print("\nS*")
# print(S_opt)

# # =====================================================
# # 制約確認
# # =====================================================

# print("\n==============================")
# print("Constraint Check")
# print("==============================")

# print("\nSupply-Demand Balance")

# for i, val in enumerate(balance_constraint(result.x)):
#     print(f"Node {i+1}: {val}")

# print("\nVoltage Constraints")

# idx = 0

# for i in range(n):
#     for j in range(i + 1, n):

#         residual = voltage_constraint(result.x)[idx]

#         lhs = (V_opt[i] - V_opt[j])**2

#         rhs = R[i, j] * (
#             S_opt[i, j] + S_opt[j, i]
#         )

#         print(
#             f"({i+1},{j+1}) "
#             f"LHS={lhs:.8f} "
#             f"RHS={rhs:.8f} "
#             f"Residual={residual:.3e}"
#         )

#         idx += 1


# # =====================================================
# # Sij + Sji の確認
# # =====================================================

# print("\n==============================")
# print("Sij + Sji")
# print("==============================")

# for i in range(n):
#     for j in range(i + 1, n):

#         value = S_opt[i, j] + S_opt[j, i]

#         print(
#             f"S{i+1}{j+1}+S{j+1}{i+1}"
#             f" = {value:.8f}"
#         )


# # import numpy as np
# # from scipy.optimize import minimize

# # #基本的なネットワークに，需給を正，ロスも正とする不等式制約を追加．

# # # =====================================
# # # パラメータ
# # # =====================================

# # n = 3

# # a = np.array([100, 110, 120])
# # b = np.array([-0.3, -0.2, -0.1])

# # c = np.array([5, 10, 15])
# # d = np.array([0.02, 0.04, 0.06])

# # # =====================================
# # # S の変数配置
# # # =====================================

# # flow_pairs = [
# #     (0, 1), (0, 2),
# #     (1, 0), (1, 2),
# #     (2, 0), (2, 1)
# # ]

# # m = len(flow_pairs)


# # # =====================================
# # # flowベクトル → S行列
# # # =====================================

# # def build_S(flow):

# #     S = np.zeros((n, n))

# #     for k, (i, j) in enumerate(flow_pairs):
# #         S[i, j] = flow[k]

# #     return S


# # # =====================================
# # # 目的関数
# # # maximize f
# # # ↓
# # # minimize -f
# # # =====================================

# # def objective(x):

# #     qd = x[:n]
# #     qs = x[n:2*n]

# #     B = a * qd + 0.5 * b * qd**2
# #     C = c * qs + 0.5 * d * qs**2

# #     welfare = np.sum(B - C)

# #     return -welfare


# # # =====================================
# # # 需給バランス制約
# # # qs_i - qd_i - ΣSij = 0
# # # =====================================

# # def balance_constraint(x):

# #     qd = x[:n]
# #     qs = x[n:2*n]

# #     flow = x[2*n:]

# #     S = build_S(flow)

# #     h = np.zeros(n)

# #     for i in range(n):
# #         h[i] = qs[i] - qd[i] - np.sum(S[i, :])

# #     return h


# # # =====================================
# # # Sij + Sji >= 0
# # # =====================================

# # def flow_constraint(x):

# #     flow = x[2*n:]

# #     S = build_S(flow)

# #     g = []

# #     for i in range(n):
# #         for j in range(i + 1, n):

# #             g.append(
# #                 S[i, j] + S[j, i]
# #             )

# #     return np.array(g)


# # # =====================================
# # # 制約
# # # =====================================

# # constraints = [
# #     {
# #         "type": "eq",
# #         "fun": balance_constraint
# #     },
# #     {
# #         "type": "ineq",
# #         "fun": flow_constraint
# #     }
# # ]


# # # =====================================
# # # bounds
# # # qd >= 0, qs >= 0
# # # Sは自由
# # # =====================================

# # bounds = []

# # # qd
# # for _ in range(n):
# #     bounds.append((0, None))

# # # qs
# # for _ in range(n):
# #     bounds.append((0, None))

# # # S
# # for _ in range(m):
# #     bounds.append((None, None))


# # # =====================================
# # # 初期値
# # # =====================================

# # x0 = np.zeros(2*n + m)

# # x0[:n] = 50.0
# # x0[n:2*n] = 50.0

# # # S初期値
# # x0[2*n:] = 0.0


# # # =====================================
# # # 最適化
# # # =====================================

# # result = minimize(
# #     objective,
# #     x0,
# #     method="SLSQP",
# #     constraints=constraints,
# #     bounds=bounds
# # )

# # # =====================================
# # # 結果表示
# # # =====================================

# # qd_opt = result.x[:n]
# # qs_opt = result.x[n:2*n]
# # flow_opt = result.x[2*n:]

# # S_opt = build_S(flow_opt)

# # print("success =", result.success)
# # print("status  =", result.status)
# # print("message =", result.message)

# # print("\nqd =", qd_opt)
# # print("qs =", qs_opt)
# # print("S =\n", S_opt)

# # # import numpy as np
# # # from scipy.optimize import minimize

# # # #最も単純なネットワーク，送電上限なしで送電がどのように決定されるかを確認する．

# # # # ------------------
# # # # パラメータ
# # # # ------------------

# # # n = 3

# # # a = np.array([100, 110, 120])
# # # b = np.array([-0.3, -0.2, -0.1])

# # # c = np.array([5, 10, 15])
# # # d = np.array([0.02, 0.04, 0.06])


# # # # ------------------
# # # # 目的関数
# # # # ------------------

# # # def objective(x):

# # #     qd = x[:n]
# # #     qs = x[n:2*n]

# # #     B = a*qd + 0.5*b*qd**2
# # #     C = c*qs + 0.5*d*qs**2

# # #     welfare = np.sum(B - C)

# # #     return -welfare    # maximize → minimize


# # # # ------------------
# # # # S行列生成
# # # # ------------------

# # # flow_pairs = [(0,1),(0,2),
# # #               (1,0),(1,2),
# # #               (2,0),(2,1)]

# # # m = len(flow_pairs)


# # # def build_S(flow):

# # #     S = np.zeros((n,n))

# # #     for k,(i,j) in enumerate(flow_pairs):
# # #         S[i,j] = flow[k]

# # #     return S


# # # # ------------------
# # # # 需給バランス制約
# # # # ------------------

# # # def balance_constraints(x):

# # #     qd = x[:n]
# # #     qs = x[n:2*n]
# # #     flow = x[2*n:]

# # #     S = build_S(flow)

# # #     h = np.zeros(n)

# # #     for i in range(n):
# # #         h[i] = qs[i] - qd[i] - np.sum(S[i,:])

# # #     return h


# # # constraints = {
# # #     'type': 'eq',
# # #     'fun': balance_constraints
# # # }


# # # # ------------------
# # # # 初期値
# # # # ------------------

# # # x0 = np.zeros(2*n + m)

# # # x0[:n] = 50      # qd
# # # x0[n:2*n] = 50  # qs


# # # # ------------------
# # # # 求解
# # # # ------------------

# # # result = minimize(
# # #     objective,
# # #     x0,
# # #     method='SLSQP',
# # #     constraints=[constraints]
# # # )

# # # # ------------------
# # # # 結果
# # # # ------------------

# # # qd = result.x[:n]
# # # qs = result.x[n:2*n]
# # # flow = result.x[2*n:]

# # # S = build_S(flow)

# # # print("success =", result.success)
# # # print()

# # # print("qd =", qd)
# # # print("qs =", qs)

# # # print()
# # # print("S =")
# # # print(S)

# # # print()
# # # print("social welfare =", -result.fun)