# %%
import numpy as np
import matplotlib.pyplot as plt
# Constrants
N_STATES = 5  
N_ITERATIONS = 50 
TOLERANCE = 1e-5  


#Construct and normalize the transition matrix P, start by creating a random 5 by 5 matrix
P = np.random.rand(N_STATES, N_STATES)

# Normalize each row so it sums to 1, divide each row by the sum of that row
row_sums = P.sum(axis=1, keepdims=True)
P = P / row_sums

print("Transition Matrix P")
print(P)
print("\nRow sums of P:", P.sum(axis=1))


# Construct a random size-5 vector p and normalize it
p_initial = np.random.rand(N_STATES)
p_initial = p_initial / p_initial.sum()

print("Initial Probability Vector p (p_0)")
print(p_initial)
print(f"Initial sum: {p_initial.sum()}")

# Apply the transition rule 50 times, use p_new = P_transpose * p_old 
p_50 = p_initial.copy()
P_T = P.T  # Transpose of P

for _ in range(N_ITERATIONS):
    p_50 = np.dot(P_T, p_50)

print(f"Vector p after {N_ITERATIONS} iterations")
print(p_50)
print(f"Sum of p_50: {p_50.sum()}")


# Compute the stationary distribution (v)
print("Stationary Distribution (v)")
# Compute eigenvalues and eigenvectors of the transpose of P
eigenvalues, eigenvectors = np.linalg.eig(P_T)

# Find the index of the eigenvalue closest to 1 
eig_1_index = np.argmin(np.abs(eigenvalues - 1.0))
print(f"Eigenvalue closest to 1: {eigenvalues[eig_1_index]}")

# Get the corresponding eigenvector, and take the real part in case it is complex
v = eigenvectors[:, eig_1_index]
v = np.real(v)

# Scale the eigenvector so its components sum to 1
v = v / v.sum()

# Take the absolute value and normalize
v = np.abs(v)
v = v / v.sum()

print("Stationary Distribution v (scaled eigenvector)")
print(v)
print(f"Sum of v: {v.sum()}")

# Compare p_50 and the stationary distribution v by computing the componet-wise diffrence
difference = np.abs(p_50 - v)

# Check if all components match within the tolerance
matches = np.all(difference < TOLERANCE)

print(f"Comparison (Tolerance = {TOLERANCE})")
print(f"p_50 = {p_50}")
print(f"v    = {v}")
print(f"\nComponent-wise difference: {difference}")
print(f"\nDo p_50 and v match within {TOLERANCE}? {matches}")

# %%



