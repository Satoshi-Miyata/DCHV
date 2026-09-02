import numpy as np
from scipy.optimize import minimize

#SとVを交互に最適化する手法．

# =====================================================
# Parameters
# =====================================================

n = 3

a = np.array([100.0, 110.0, 120.0])
b = np.array([-0.3, -0.2, -0.1])

c = np.array([5.0, 10.0, 15.0])
d = np.array([0.02, 0.04, 0.06])

rho = 10e-2
eps = 0.01

P_min = -100.0
P_max = 100.0

R = np.array([
    [0.0, 0.2, 0.1],
    [0.2, 0.0, 0.5],
    [0.1, 0.5, 0.0]
])

max_iter = 20

# =====================================================
# Flow indices
# =====================================================

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
# Utilities
#潮流Sの箱を定義
# =====================================================

def build_S(flow_vec):

    S = np.zeros((n, n))

    for k, (i, j) in enumerate(flow_pairs):
        S[i, j] = flow_vec[k]

    return S

#最適化変数xを分割する関数
def split_main(x):

    qd = x[:n]
    qs = x[n:2*n]
    s = x[2*n:]

    return qd, qs, build_S(s)


# =====================================================
# Welfare
# =====================================================

def welfare(qd, qs):

    B = a * qd + 0.5 * b * qd**2
    C = c * qs + 0.5 * d * qs**2

    return np.sum(B - C)

# =====================================================
# Main problem
# =====================================================

def solve_main_problem(S_hat_prev):

    def obj(x):

        qd, qs, S = split_main(x)

        penalty = np.sum(
            (S - S_hat_prev)**2
        )

        value = welfare(qd, qs) - rho * penalty

        return -value

    def balance(x):

        qd, qs, S = split_main(x)

        h = np.zeros(n)

        for i in range(n):

            flow_sum = np.sum(S[i, :])

            h[i] = qs[i] - qd[i] - flow_sum

        return h

    def physical_constraint(x):

        qd, qs, S = split_main(x)

        g = []

        for i in range(n):
            for j in range(n):

                g.append( S[i,j] + S[j,i] )
        
        return np.array(g)

    x0 = np.concatenate([
        np.ones(n) * 10.0,
        np.ones(n) * 10.0,
        np.random.rand(n**2 - n)
        # np.zeros(n**2 - n)
    ])

    constraints = [
        {
            "type": "eq",
            "fun": balance
        },
        {
            "type": "eq",
            "fun": physical_constraint
        }
    ]

    bounds = []

    # qd >= 0
    for _ in range(n):
        bounds.append((0, None))

    # qs >= 0
    for _ in range(n):
        bounds.append((0, None))

    # flow bounds
    for _ in range(m):
        bounds.append((P_min, P_max))

    result = minimize(
        obj,
        x0,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints
    )
    print("objective =", obj(x0))
    print("balance   =", balance(x0))
    print("physical  =", physical_constraint(x0))

    return result

# =====================================================
# Voltage identification
#最適な電圧を求める関数
# =====================================================

def identify_voltage(S):

    def J(V):

        value = 0.0

        for i in range(n):
            for j in range(i + 1, n):

                target = ( R[i, j] * (S[i, j] + S[j, i]) )

                value += ( (V[i] - V[j])**2 - target )**2

        return value

    result = minimize(
        J,
        # x0=np.ones(n),
        x0=np.random.rand(n),
        method="BFGS"
    )
    print("Voltage identification result:")
    print("収束確認:", result.success)
    print("関数値:", result.fun)
    print("最適化変数:", result.x)
    print("---------------------------------")

    return result.x


# =====================================================
# Compute physical flow
#電圧から潮流を計算する関数
# =====================================================

def physical_flow(V):

    S_hat = np.zeros((n, n))

    for i in range(n):
        for j in range(n):

            if i == j:
                continue

            S_hat[i, j] = ( V[i]**2 - V[i]*V[j] ) / R[i, j]

    return S_hat

# =====================================================
# Initial S_hat^(0)
# from V^(0)=(1,1,1)
# =====================================================

V0 = np.random.rand(n)
print("Initial V0 =", V0)

S_hat_prev = physical_flow(V0)
print("Initial S_hat =\n", S_hat_prev)


# =====================================================
# Iteration
# =====================================================

for k in range(1, max_iter + 1):

    print(f"\n========== Iteration {k} ==========")

    # Step 1
    main_result = solve_main_problem(S_hat_prev)

    qd, qs, S = split_main(main_result.x)
    print("qd =", qd, "qs =", qs, "\nS =\n", S)
    print("S+S.T =\n", S + S.T)
    print("min(S+S.T) =", np.min(S + S.T))
    print("main_result.success =", main_result.success)
    print("main_result.message =", main_result.message)
    print("main_result.fun =", main_result.fun)
    print("main_result.status =", main_result.status)

    # Step 2
    V = identify_voltage(S)
    print("V =", V)

    # Step 3
    S_hat = physical_flow(V)
    print("S_hat =\n", S_hat)

    # Step 4
    error = np.max(np.abs(S - S_hat))

    print("max error =", error)

    if error < eps:

        print("\nConverged")

        print("\nqd")
        print(qd)

        print("\nqs")
        print(qs)

        print("\nS")
        print(S)

        print("\nV")
        print(V)

        print("\nS_hat")
        print(S_hat)

        print("\nSocial welfare")
        print(welfare(qd, qs))

        break

    S_hat_prev = S_hat

else:

    print("\nMaximum iteration reached.")