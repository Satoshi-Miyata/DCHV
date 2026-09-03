import numpy as np
import matplotlib.pyplot as plt
import sys
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

rho = 10e2
eps = 0.01

P_min = -100.0
P_max = 100.0

R = np.array([
    [0.0, 0.2, 0.1],
    [0.2, 0.0, 0.5],
    [0.1, 0.5, 0.0]
])

max_iter = 10

error_history = []

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
            for j in range(i+1, n):

                g.append( S[i,j] + S[j,i] )
        
        return np.array(g)
    
    # =====================================================
    # 初期値の設定:S,qd,qs
    # =====================================================
    #Sとqdを設定, qsはqd+sum(S)で初期化．初期値が等式を満足するようにするため．

    check_initial_S0 = 0

    # #ランダム-------------------------------------------------
    # S0_vec = np.random.uniform(0, 1, m)
    # S0 = build_S(S0_vec)
    # qd0 = np.ones(n) * 10.0
    # check_initial_S0 += 1
    # #-------------------------------------------------

    #定数---------------------------------------------
    S0_vec = np.ones(m)
    S0 = build_S(S0_vec)
    qd0 = np.ones(n) * 10.0
    check_initial_S0 += 1
    #-------------------------------------------------

    if check_initial_S0 > 1:
        print("S0とqdの初期値の設定が複数あります.")
        return sys.exit()
    
    elif check_initial_S0 == 0:
        print("S0とqdの初期値が設定されていません.")
        return sys.exit()

    qs0 = np.zeros(n)
    for i in range(n):
        qs0[i] = qd0[i] + np.sum(S0[i, :])

    x0 = np.concatenate([
        qd0,
        qs0,
        S0_vec
    ])

    constraints = [
        {
            "type": "eq",
            "fun": balance
        },
        {
            "type": "ineq",
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
    # print("Initial objective =", obj(x0))
    # print("Initial balance   =", balance(x0))
    # print("Initial physical  =", physical_constraint(x0))

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

    return result


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
# 初期値の設定:V0,S_hat_prev
# =====================================================

check_initial_V0 = 0

#ランダム------------------------------------------
V0 = 100 * np.random.rand(n)
check_initial_V0 += 1
#-------------------------------------------------

# #任意の定数なV0------------------------------------
# V0 = np.ones(n)
# check_initial_V0 += 2
# #-------------------------------------------------

if check_initial_V0 > 2:
    print("V0の初期値の設定が複数あります.")
    sys.exit()

elif check_initial_V0 == 0:
    print("V0の初期値が設定されていません.")
    sys.exit()

S_hat_prev = physical_flow(V0)

if check_initial_V0 % 2 == 1:
    print("初期値の電圧V0をランダムに生成しました。")
else:
    print("初期値の電圧V0を定数で設定しました。")
print("Initial V0 =", V0)
print("Initial S_hat =\n", S_hat_prev)


# =====================================================
# Iteration
# =====================================================

for k in range(1, max_iter + 1):

    # Step 1
    main_result = solve_main_problem(S_hat_prev)
    qd, qs, S = split_main(main_result.x)

    # Step 2
    result_V = identify_voltage(S)
    V = result_V.x

    # Step 3
    S_hat = physical_flow(V)

    # Step 4
    error = np.max(np.abs(S - S_hat))
    error_history.append(error)


    print(f"\n========== Iteration {k} ==========")
    # print("==Main problem result:")
    # print("qd =", qd, "\nqs =", qs, "\nS =\n", S)
    # print("S_hat =\n", S_hat)
    # print("S+S.T =\n", S + S.T)
    # print("min(S+S.T) =", np.min(S + S.T), "\nmax(S+S.T) =", np.max(S + S.T))
    # print("計算成否:", main_result.success)
    # print("社会厚生:", -1 * main_result.fun)
    # # print("main_result.message =", main_result.message)
    # # print("main_result.status =", main_result.status)
    # print("---------------------------------")
    # print("==Voltage identification result:")
    print("V =", result_V.x)
    print("V1-V2 =", V[0] - V[1], "\nV1-V3 =", V[0] - V[2], "\nV2-V3 =", V[1] - V[2])
    # print("計算成否:", result_V.success)
    # print("関数値:", result_V.fun)
    # print("---------------------------------")
    # print("max error =", error)

    if error < eps:
        #epsのalpha倍以下の誤差が何回出たかを数える
        alpha = 5
        near_count = np.sum(
            np.array(error_history) <= alpha * eps
        )
        first_near = np.where(
            np.array(error_history) <= alpha * eps
        )[0][0]

        print("\n====Converged: iteration =", k, "====")
        print("収束性評価", near_count, "回")
        print("最初に", alpha,"eps近傍に入ったのは", first_near,"回目")

        print("qd = ", qd)
        print("qs = ", qs)
        print("S = ")
        print(S)
        print("min(S+S.T) =", np.min(S + S.T), "\nmax(S+S.T) =", np.max(S + S.T))

        print("V = ", V)
        print("計算可否:", result_V.success)
        print("関数値:", result_V.fun)

        # print("\nS_hat")
        # print(S_hat)

        print("\nSocial welfare")
        print(welfare(qd, qs))

        break

    S_hat_prev = S_hat

else:

    print("\nMaximum iteration reached.")


plt.figure(figsize=(8,5))

plt.plot(
    range(1, len(error_history)+1),
    error_history,
    marker='o'
)

plt.xlabel("Iteration")
plt.ylabel("Max Error")
plt.title("Convergence History")
plt.grid(True)

plt.show()
