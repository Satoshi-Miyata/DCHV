import numpy as np
from scipy.optimize import minimize

#交互最適する前に，一括最適化できるか検証．S=f(V)の潮流式を制約に追加

# =====================================================
# Parameters
# =====================================================

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

V_base = 100.0

# =====================================================
# S の管理
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


def build_S(flow):

    S = np.zeros((n, n))

    for k, (i, j) in enumerate(flow_pairs):
        S[i, j] = flow[k]

    return S


# =====================================================
# Objective
# maximize welfare
# ↓
# minimize (-welfare)
# =====================================================

def objective(x):

    qd = x[:n]
    qs = x[n:2*n]

    B = a * qd + 0.5 * b * qd**2
    C = c * qs + 0.5 * d * qs**2

    welfare = np.sum(B - C)

    return -welfare


# =====================================================
# Supply-demand balance
# =====================================================

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


# =====================================================
# Power flow constraint
# Sij = Vi(Vi-Vj)/Rij
# =====================================================

def power_flow_constraint(x):

    flow = x[2*n:2*n+m]

    V = x[2*n+m:]

    S = build_S(flow)

    residual = []

    for i, j in flow_pairs:

        residual.append(
            S[i, j]
            - V[i] * (V[i] - V[j]) / R[i, j]
        )

    return np.array(residual)


# =====================================================
# Reference voltage
# V1 = Vbase
# =====================================================

def reference_voltage_constraint(x):

    V = x[2*n+m:]

    return np.array([
        V[0] - V_base
    ])


# =====================================================
# Constraints
# =====================================================

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
        "fun": reference_voltage_constraint
    }
]

# =====================================================
# Bounds
# =====================================================

bounds = []

# qd >= 0

for _ in range(n):
    bounds.append((0.0, None))

# qs >= 0

for _ in range(n):
    bounds.append((0.0, None))

# -100 <= Sij <= 100

for _ in range(m):
    bounds.append((-100.0, 100.0))

# Voltage : no bounds (Case A)

for _ in range(n):
    bounds.append((None, None))

# =====================================================
# Initial value
# =====================================================

x0 = np.zeros(2*n + m + n)

# qd
x0[:n] = 100

# qs
x0[n:2*n] = 100

# S
x0[2*n:2*n+m] = 0

# V
# x0[2*n+m:] = 1
x0[2*n+m:] = np.random.uniform(-10, 10, n)

# =====================================================
# Solve
# =====================================================

result = minimize(
    objective,
    x0,
    method="SLSQP",
    bounds=bounds,
    constraints=constraints,
    options={
        "maxiter": 2000,
        "ftol": 1e-9,
        "disp": True
    }
)

# =====================================================
# Extract
# =====================================================

qd_opt = result.x[:n]

qs_opt = result.x[n:2*n]

flow_opt = result.x[2*n:2*n+m]

V_opt = result.x[2*n+m:]

S_opt = build_S(flow_opt)

# =====================================================
# Output
# =====================================================
print("\nInitial_V :", x0[2*n+m:])

print("\n====================")
print("Optimization Result")
print("====================")

print("Success :", result.success)
print("Status  :", result.status)
print("Message :", result.message)

print("\nMaximum Welfare")
print(-result.fun)

print("\nqd*")
print(qd_opt)

print("\nqs*")
print(qs_opt)

print("\nV*")
print(V_opt)

print("\nS*")
print(S_opt)

print("\nLoss")
print(S_opt + S_opt.T)

print("\n====================")
print("Constraint Check")
print("====================")

print("\nBalance residual")
print(balance_constraint(result.x))

print("\nS定義式の制約:Power flow residual")
print(power_flow_constraint(result.x))

print("\nVoltage reference residual")
print(reference_voltage_constraint(result.x))

print("\n====================")
print("Branch Information")
print("====================")

for i, j in flow_pairs:

    lhs = S_opt[i, j]

    rhs = (
        V_opt[i]
        * (V_opt[i] - V_opt[j])
        / R[i, j]
    )

    print(
        f"S{i+1}{j+1}: "
        f"{lhs:.6f} "
        f"(theory={rhs:.6f}, "
        f"error={lhs-rhs:.3e})"
    )
