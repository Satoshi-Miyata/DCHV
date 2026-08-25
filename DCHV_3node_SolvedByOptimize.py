import numpy as np
from scipy.optimize import minimize, NonlinearConstraint, Bounds

# =====================================================
# Parameters
# =====================================================

n = 3

a = np.array([100.0, 110.0, 120.0])
b = np.array([-0.3, -0.2, -0.1])

c = np.array([5.0, 10.0, 15.0])
d = np.array([0.02, 0.04, 0.06])

R = np.array([
    [0.0, 2.0, 1.0],
    [2.0, 0.0, 5.0],
    [1.0, 5.0, 0.0]
])

P_lower = -100.0
P_upper = 100.0

V_ref = 40

# =====================================================
# Variable decomposition
# x = [qd1,qd2,qd3, qs1,qs2,qs3, V1,V2,V3]
# =====================================================

def split_x(x):

    qd = x[0:n]

    qs = x[n:2*n]

    V = x[2*n:3*n]

    return qd, qs, V

# =====================================================
# Objective
# maximize f
# →
# minimize -f
# =====================================================

def objective(x):

    qd, qs, V = split_x(x)

    benefit = np.sum(
        a * qd +
        0.5 * b * qd**2
    )

    cost = np.sum(
        c * qs +
        0.5 * d * qs**2
    )

    f = benefit - cost

    return -f

# =====================================================
# h(qd,qs,V)
# =====================================================

def h_constraint(x):

    qd, qs, V = split_x(x)

    h = np.zeros(n)

    for i in range(n):

        power_flow = 0.0

        for j in range(n):

            if i == j:
                continue

            power_flow += (
                V[i] *
                (V[i] - V[j])
                / R[i, j]
            )

        h[i] = (
            qs[i]
            - qd[i]
            - power_flow
        )

    return h

# =====================================================
# V_ref_constraint
# =====================================================
def vref_constraint(x):

    qd, qs, V = split_x(x)

    return V[0] - V_ref

# =====================================================
# g-(V)
# =====================================================

def g_minus_constraint(x):

    qd, qs, V = split_x(x)

    values = []

    for i in range(n):
        for j in range(n):

            if i == j:
                continue

            pij = (
                V[i]**2
                - V[i]*V[j]
            ) / R[i, j]

            values.append(
                pij - P_lower
            )

    return np.array(values)

# =====================================================
# g+(V)
# =====================================================

def g_plus_constraint(x):

    qd, qs, V = split_x(x)

    values = []

    for i in range(n):
        for j in range(n):

            if i == j:
                continue

            pij = (
                V[i]**2
                - V[i]*V[j]
            ) / R[i, j]

            values.append(
                P_upper - pij
            )

    return np.array(values)

# =====================================================
# Constraints
# =====================================================

h_nlc = NonlinearConstraint(
    h_constraint,
    0.0,
    0.0
)

vref_nlc = NonlinearConstraint(
    vref_constraint,
    0.0,
    0.0
)

gminus_nlc = NonlinearConstraint(
    g_minus_constraint,
    0.0,
    np.inf
)

gplus_nlc = NonlinearConstraint(
    g_plus_constraint,
    0.0,
    np.inf
)

# =====================================================
# Bounds
#
# qd >= 0
# qs >= 0
# V >= 0
# =====================================================

lower = np.zeros(3*n)

upper = np.full(3*n, np.inf)

bounds = Bounds(lower, upper)

# =====================================================
# Initial point
# =====================================================

x0 = np.array([
    100, 100, 100,
    100, 100, 100,
    1, 1, 1
], dtype=float)

# =====================================================
# Solve
# =====================================================

result = minimize(
    objective,
    x0,
    method="trust-constr",
    constraints=[
        h_nlc,
        vref_nlc,
        gminus_nlc,
        gplus_nlc
    ],
    bounds=bounds,
    options={
        "verbose": 3,
        "maxiter": 3000
    }
)

# =====================================================
# Result
# =====================================================

qd, qs, V = split_x(result.x)

P = np.zeros((n, n))

for i in range(n):
    for j in range(n):
        if i != j:
            P[i, j] = (V[i]**2 - V[i] * V[j]) / R[i, j]



print("\n======================")
print("Optimization Result")
print("======================")

print("\nSuccess:")
print(result.success)

print("\nMessage:")
print(result.message)

print("\nObjective value:")
print(-result.fun)

print("\nqd*")
print(qd)

print("\nqs*")
print(qs)

print("\nDCgap = Loss")
print(np.sum(qd)-np.sum(qs))

print("\nV*")
print(V)

print("\nh(q*,V*)")
print(h_constraint(result.x))

print("\ng-(V*)")
print(g_minus_constraint(result.x))

print("\ng+(V*)")
print(g_plus_constraint(result.x))

print("P =")
print(P)