# Import necessary libraries
import numpy as np
import tensorly as tl
from tensorly.decomposition import parafac

# Example: Creating a 2D OD matrix with travel times
O, D = 50, 40  # Dimensions for Origins (O) and Destinations (D)
np.random.seed(42)  # For reproducibility
OD_matrix = np.random.rand(O, D)  # Random travel time data

# Convert the 2D matrix into a tensor format
tensor = tl.tensor(OD_matrix)

# Define rank for CP decomposition
rank = 4  # Adjust as needed

# Perform CP decomposition
weights, factors = parafac(tensor, rank=rank, return_errors=False)

# Extract the factor matrices
origin_factors = factors[0]  # Factor matrix for Origins
destination_factors = factors[1]  # Factor matrix for Destinations

# Print the original matrix
print("Original OD Matrix:")
print(OD_matrix)

# Loop through each rank component to print details
print("\nDetailed Rank Components:")
for r in range(rank):
    print(f"\nRank-{r + 1} Component:")
    print(f"Weights: {weights[r]:.4f}")
    print("Origin Factors (O):")
    print(origin_factors[:, r])
    print("Destination Factors (D):")
    print(destination_factors[:, r])

    # Outer product for this rank
    component = weights[r] * np.outer(origin_factors[:, r], destination_factors[:, r])
    print("Outer Product of Rank Component:")
    print(component)

print("\nFactor Matrix for Origins (O):")
print(origin_factors)
print("\nFactor Matrix for Destinations (D):")
print(destination_factors)

# Reconstruct the matrix using all components
reconstructed_matrix = np.zeros_like(OD_matrix)
for r in range(rank):
    reconstructed_matrix += weights[r] * np.outer(origin_factors[:, r], destination_factors[:, r])

# Print the reconstructed matrix
print("\nReconstructed OD Matrix (Approximation):")
print(reconstructed_matrix)

# Compute reconstruction error
reconstruction_error = np.linalg.norm(OD_matrix - reconstructed_matrix) / np.linalg.norm(OD_matrix)
print(f"\nReconstruction Error: {reconstruction_error:.4f}")