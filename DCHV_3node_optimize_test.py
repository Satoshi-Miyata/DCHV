import numpy as np
from scipy.optimize import minimize

# =====================================================
# Parameters
# =====================================================

n = 3

a = np.array([100.0, 110.0, 120.0])
b = np.array([-0.3, -0.2, -0.1])

c = np.array([5.0, 10.0, 15.0])
d = np.array([0.02, 0.04, 0.06])

P_min = -100.0
P_max = 100.0

R = np.array([
    [0.0, 0.2, 0.1],
    [0.2, 0.0, 0.5],
    [0.1, 0.5, 0.0]
])

# =====================================================
# x = [qd1 qd2 qd3 qs1 qs2 qs3 V1 V2 V3]
# =====================================================

def split_x(x):

    qd = x[0:n]
    qs = x[n:2*n]
    V  = x[2*n:3*n]

    return qd, qs, V

# =====================================================
# Utility / Cost
# =====================================================

def objective(x):

    qd, qs, V = split_x(x)

    B = a * qd + 0.5 * b * qd**2
    C = c * qs + 0.5 * d * qs**2

    welfare = np.sum(B - C)

    return -welfare   # maximize -> minimize

# =====================================================
# Power flow
# =====================================================

def Pij(V, i, j):

    return (V[i]**2 - V[i]*V[j]) / R[i, j]

# =====================================================
# Equality constraints
# h_i = 0
# =====================================================

def balance_constraint(x):

    qd, qs, V = split_x(x)

    h = np.zeros(n)

    for i in range(n):

        flow_sum = 0.0

        for j in range(n):

            if i == j:
                continue

            flow_sum += Pij(V, i, j)

        h[i] = qs[i] - qd[i] - flow_sum

    return h

# =====================================================
# Lower flow constraint
# Pij - Pmin >= 0
# =====================================================

def lower_flow_constraint(x):

    _, _, V = split_x(x)

    g = []

    for i in range(n):
        for j in range(n):

            if i == j:
                continue

            g.append(Pij(V, i, j) - P_min)

    return np.array(g)

# =====================================================
# Upper flow constraint
# Pmax - Pij >= 0
# =====================================================

def upper_flow_constraint(x):

    _, _, V = split_x(x)

    g = []

    for i in range(n):
        for j in range(n):

            if i == j:
                continue

            g.append(P_max - Pij(V, i, j))

    return np.array(g)

# =====================================================
# Initial point
# =====================================================

qd0 = np.array([10.0, 10.0, 10.0])
qs0 = np.array([10.0, 10.0, 10.0])

V0 = np.array([1.0, 1.0, 1.0])

x0 = np.concatenate([qd0, qs0, V0])

# =====================================================
# Constraints
# =====================================================

constraints = [

    {
        "type": "eq",
        "fun": balance_constraint
    },

    {
        "type": "ineq",
        "fun": lower_flow_constraint
    },

    {
        "type": "ineq",
        "fun": upper_flow_constraint
    }

]

# =====================================================
# Bounds
# qd >= 0, qs >= 0
# Vは無制約
# =====================================================

bounds = []

# qd
for _ in range(n):
    bounds.append((0, None))

# qs
for _ in range(n):
    bounds.append((0, None))

# V (unbounded)
for _ in range(n):
    bounds.append((None, None))

# =====================================================
# Optimization
# =====================================================

result = minimize(
    objective,
    x0,
    method="SLSQP",
    bounds=bounds,
    constraints=constraints,
    options={
        "maxiter": 1000,
        "disp": True
    }
)

# =====================================================
# Results
# =====================================================

qd, qs, V = split_x(result.x)

print("\n========================")
print("Optimization Result")
print("========================")

print("Success =", result.success)
print("Message =", result.message)

print("\nqd")
print(qd)

print("\nqs")
print(qs)

print("\nV")
print(V)

print("\nPower Flow")

for i in range(n):
    for j in range(n):

        if i == j:
            continue
        else:
            print((V[i]**2 - V[i]*V[j]) / R[i, j])

      