import numpy as np

# ==========================
# 適当なテストデータ
# ==========================

p = np.array([2.0, 3.0, 5.0])

V = np.array([1.5, -0.8, 2.1])

R = np.array([
    [0.0, 0.2, 0.1],
    [0.2, 0.0, 0.5],
    [0.1, 0.5, 0.0]
])

pi_minus = np.array([
    [0.0, 2.0, 3.0],
    [4.0, 0.0, 5.0],
    [6.0, 7.0, 0.0]
])

pi_plus = np.array([
    [0.0, 0.5, 1.0],
    [1.5, 0.0, 2.0],
    [2.5, 3.0, 0.0]
])

# ==========================
# 元の総和表記
# ==========================

def original_expression(V, p, pi_minus, pi_plus, R):

    F = np.zeros(3)

    for i in range(3):

        term1 = 0.0
        for m in range(3):
            if m == i:
                continue
            term1 += p[m] * V[m] / R[m, i]

        term2 = 0.0
        for l in range(3):
            if l == i:
                continue
            term2 += (2 * V[i] - V[l]) / R[i, l]

        term2 *= p[i]

        term3 = 0.0
        for j in range(3):
            if j == i:
                continue

            delta_pi = pi_minus[i, j] - pi_plus[i, j]

            term3 += (
                delta_pi
                * (2 * V[i] - V[j])
                / R[i, j]
            )

        term4 = 0.0
        for k in range(3):
            if k == i:
                continue

            delta_pi = pi_minus[k, i] - pi_plus[k, i]

            term4 += (
                delta_pi
                * V[k]
                / R[k, i]
            )

        F[i] = term1 - term2 + term3 - term4

    return F

# ==========================
# A行列
# ==========================

def build_A(p, pi_minus, pi_plus, R):

    dp = pi_minus - pi_plus

    A = np.zeros((3, 3))

    A[0, 0] = (
        -2*p[0]/R[0,1]
        -2*p[0]/R[0,2]
        +2*dp[0,1]/R[0,1]
        +2*dp[0,2]/R[0,2]
    )

    A[0, 1] = (
        p[1]/R[1,0]
        +p[0]/R[0,1]
        -dp[0,1]/R[0,1]
        -dp[1,0]/R[1,0]
    )

    A[0, 2] = (
        p[2]/R[2,0]
        +p[0]/R[0,2]
        -dp[0,2]/R[0,2]
        -dp[2,0]/R[2,0]
    )

    A[1, 0] = (
        p[0]/R[0,1]
        +p[1]/R[1,0]
        -dp[1,0]/R[1,0]
        -dp[0,1]/R[0,1]
    )

    A[1, 1] = (
        -2*p[1]/R[1,0]
        -2*p[1]/R[1,2]
        +2*dp[1,0]/R[1,0]
        +2*dp[1,2]/R[1,2]
    )

    A[1, 2] = (
        p[2]/R[2,1]
        +p[1]/R[1,2]
        -dp[1,2]/R[1,2]
        -dp[2,1]/R[2,1]
    )

    A[2, 0] = (
        p[0]/R[0,2]
        +p[2]/R[2,0]
        -dp[2,0]/R[2,0]
        -dp[0,2]/R[0,2]
    )

    A[2, 1] = (
        p[1]/R[1,2]
        +p[2]/R[2,1]
        -dp[2,1]/R[2,1]
        -dp[1,2]/R[1,2]
    )

    A[2, 2] = (
        -2*p[2]/R[2,0]
        -2*p[2]/R[2,1]
        +2*dp[2,0]/R[2,0]
        +2*dp[2,1]/R[2,1]
    )

    return A

# ==========================
# 比較
# ==========================

F_original = original_expression(
    V, p, pi_minus, pi_plus, R
)

A = build_A(
    p, pi_minus, pi_plus, R
)

F_matrix = A @ V

# ==========================
# 固有値
# ==========================
eigvals = np.linalg.eigvals(A)

print("元の式")
print(F_original)

print("\nAV")
print(F_matrix)

print("\n差")
print(F_original - F_matrix)

print("\n最大誤差")
print(np.max(np.abs(F_original - F_matrix)))

print("\n行列式")
print(np.linalg.det(A))

print("\nランク")
print(np.linalg.matrix_rank(A))

print("\n固有値")
print(eigvals)