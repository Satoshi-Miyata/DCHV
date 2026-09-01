import numpy as np
from scipy.optimize import minimize
from scipy.optimize import least_squares

# =====================================================
# Parameters
# =====================================================

n = 3

a = np.array([100, 110, 120])
b = np.array([-0.3, -0.2, -0.1])

c = np.array([5, 10, 15])
d = np.array([0.02, 0.04, 0.06])

lambda_loss = 0.1

P_min = -100
P_max = 100

# =====================================================
# Flow variables
# =====================================================

flow_pairs = [
    (0, 1),  # S12
    (0, 2),  # S13
    (1, 0),  # S21
    (1, 2),  # S23
    (2, 0),  # S31
    (2, 1)   # S32
]

num_flows = len(flow_pairs)

# =====================================================
# Variable decomposition
# =====================================================

def split_x(x):

    qd = x[:n]

    qs = x[n:2*n]

    flows = x[2*n:]

    S = np.zeros((n, n))

    for k, (i, j) in enumerate(flow_pairs):
        S[i, j] = flows[k]

    return qd, qs, S

# =====================================================
# Loss
# =====================================================

def total_loss(S):

    loss = 0.0

    for i in range(n):
        for j in range(i + 1, n):

            Lij = S[i, j] + S[j, i]

            loss += Lij

    return loss

# =====================================================
# Objective
# =====================================================

def objective(x):

    qd, qs, S = split_x(x)

    B = a * qd + 0.5 * b * qd**2
    C = c * qs + 0.5 * d * qs**2

    welfare = np.sum(B - C)

    loss = total_loss(S)

    value = welfare - lambda_loss * loss

    return -value

# =====================================================
# Power balance constraints
# =====================================================

def balance_constraint(x):

    qd, qs, S = split_x(x)

    h = np.zeros(n)

    for i in range(n):

        flow_sum = np.sum(S[i, :])

        h[i] = qs[i] - qd[i] - flow_sum

    return h

# =====================================================
# Loss nonnegative constraints
# Lij = Sij + Sji >= 0
# =====================================================

def loss_nonnegative_constraint(x):

    _, _, S = split_x(x)

    g = []

    for i in range(n):
        for j in range(i + 1, n):

            g.append(S[i, j] + S[j, i])

    return np.array(g)

# =====================================================
# Initial point
# =====================================================

qd0 = np.array([10.0, 10.0, 10.0])
qs0 = np.array([10.0, 10.0, 10.0])

S0 = np.zeros(num_flows)

x0 = np.concatenate([qd0, qs0, S0])

# =====================================================
# Constraints
# =====================================================

constraints = [

    {
        'type': 'eq',
        'fun': balance_constraint
    },

    {
        'type': 'ineq',
        'fun': loss_nonnegative_constraint
    }

]

# =====================================================
# Bounds
# =====================================================

bounds = []

# qd >= 0
for _ in range(n):
    bounds.append((0, None))

# qs >= 0
for _ in range(n):
    bounds.append((0, None))

# -100 <= Sij <= 100
for _ in range(num_flows):
    bounds.append((P_min, P_max))

# =====================================================
# Solve
# =====================================================

result = minimize(
    objective,
    x0,
    method='SLSQP',
    bounds=bounds,
    constraints=constraints
)

# =====================================================
# Results
# =====================================================

qd, qs, S = split_x(result.x)

print("Success =", result.success)
print("Message =", result.message)

print("\nqd")
print(qd)

print("\nqs")
print(qs)

print("\nS")
print(S)

print("\nLij")

for i in range(n):
    for j in range(i + 1, n):

        print(
            f"L{i+1}{j+1} = "
            f"{S[i,j] + S[j,i]}"
        )

print("\nTotal Loss")
print(total_loss(S))

print("\nSocial Welfare")

B = a * qd + 0.5 * b * qd**2
C = c * qs + 0.5 * d * qs**2

print(np.sum(B - C))

print("\nObjective Value")
print(-result.fun)

# ==================================================
# 以下でSからVを求める
# ==================================================


# ==================================================
# Resistance matrix
# ==================================================

R = np.array([
    [0.0, 0.2, 0.1],
    [0.2, 0.0, 0.5],
    [0.1, 0.5, 0.0]
])

# ==================================================
# Residual equations
# ==================================================

def residuals(V):

    V1, V2, V3 = V

    res = []

    for i in range(3):
        for j in range(3):

            if i == j:
                continue

            model = (V[i]**2 - V[i]*V[j]) / R[i, j]

            res.append(
                model - S[i, j]
            )

    return np.array(res)

# ==================================================
# Initial guess
# ==================================================

V0 = np.array([1.0, 1.0, 1.0])

# ==================================================
# Solve
# ==================================================

result = least_squares(
    residuals,
    V0
)

# ==================================================
# Check consistency
# ==================================================

tol = 1e-6

max_residual = np.max(
    np.abs(
        residuals(result.x)
    )
)

print("===== Result =====")

if max_residual < tol:

    print("解あり")

    print("\nVoltage")

    for i, v in enumerate(result.x):
        print(f"V{i+1} = {v:.10f}")

    print("\nMaximum residual")
    print(max_residual)

else:

    print("解なし")

    print("\nBest-fit voltage")

    for i, v in enumerate(result.x):
        print(f"V{i+1} = {v:.10f}")

    print("\nMaximum residual")
    print(max_residual)