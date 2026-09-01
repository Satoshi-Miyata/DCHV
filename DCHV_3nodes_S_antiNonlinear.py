import numpy as np
from scipy.optimize import minimize

# =========================
# parameters
# =========================

n = 3

a = np.array([100, 110, 120])
b = np.array([-0.3, -0.2, -0.1])

c = np.array([5, 10, 15])
d = np.array([0.02, 0.04, 0.06])

P_min = -100
P_max = 100

# =========================
# flow index
# =========================

flow_pairs = [
    (0, 1),  # S12
    (0, 2),  # S13
    (1, 0),  # S21
    (1, 2),  # S23
    (2, 0),  # S31
    (2, 1)   # S32
]

num_flows = len(flow_pairs)

# =========================
# split variables
# =========================

def split_x(x):

    qd = x[:n]

    qs = x[n:2*n]

    flow_vec = x[2*n:]

    S = np.zeros((n, n))

    for k, (i, j) in enumerate(flow_pairs):
        S[i, j] = flow_vec[k]

    return qd, qs, S

# =========================
# objective function
# =========================

def objective(x):

    qd, qs, _ = split_x(x)

    B = a * qd + 0.5 * b * qd**2
    C = c * qs + 0.5 * d * qs**2

    welfare = np.sum(B - C)

    return -welfare      # maximize -> minimize

# =========================
# equality constraints
# =========================

def balance_constraint(x):

    qd, qs, S = split_x(x)

    h = np.zeros(n)

    for i in range(n):

        flow_sum = np.sum(S[i, :])

        h[i] = qs[i] - qd[i] - flow_sum

    return h

# =========================
# initial point
# =========================

qd0 = np.ones(n) * 10.0
qs0 = np.ones(n) * 10.0

S0 = np.zeros(num_flows)

x0 = np.concatenate([
    qd0,
    qs0,
    S0
])

# =========================
# constraints
# =========================

constraints = [
    {
        'type': 'eq',
        'fun': balance_constraint
    }
]

# =========================
# bounds
# =========================

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

# =========================
# solve
# =========================

result = minimize(
    objective,
    x0,
    method="SLSQP",
    bounds=bounds,
    constraints=constraints
)

# =========================
# output
# =========================

print("Success:", result.success)
print("Message:", result.message)

qd, qs, S = split_x(result.x)

print("\nqd")
print(qd)

print("\nqs")
print(qs)

print("\nS")
print(S)

print("\n Supply - Demand")
print(np.sum(qs) - np.sum(qd))

print("\nObjective value")
print(-result.fun)