import torch
import torch.nn as nn
import numpy as np

# Example: Creating a 2D OD matrix with travel times
O, D = 50, 40  # Dimensions for Origins (O) and Destinations (D)
np.random.seed(42)  # For reproducibility
OD_matrix = np.random.rand(O, D)  # Random travel time data
tensor = torch.tensor(OD_matrix, dtype=torch.float32)  # Convert to PyTorch tensor

# Define rank for CP decomposition
rank = 4  # Adjust as needed

# Example covariates for origins (e.g., socioeconomic data)
covariates = torch.rand(O, 3)  # 3 covariates for each origin

# Define nonlinear model for generating origin factors from covariates
class NonlinearMapping(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(NonlinearMapping, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 16),  # Hidden layer
            nn.ReLU(),                # Nonlinear activation
            nn.Linear(16, output_dim) # Output layer (maps to rank dimension)
        )

    def forward(self, x):
        return self.model(x)

# Initialize the nonlinear mapping model
origin_model = NonlinearMapping(input_dim=3, output_dim=rank)  # Map covariates to rank-dimensional factors
destination_factors = torch.rand(D, rank, requires_grad=True)  # Randomly initialize destination factors
weights = torch.ones(rank, requires_grad=True)  # Initialize weights for each rank component

# Define optimizer
optimizer = torch.optim.Adam(
    list(origin_model.parameters()) + [destination_factors, weights], lr=0.01
)

# Training loop
num_iterations = 1000
for iteration in range(num_iterations):
    optimizer.zero_grad()

    # Compute origin factors using the nonlinear model
    origin_factors = origin_model(covariates)

    # Reconstruct the OD matrix using CP decomposition
    reconstructed_matrix = torch.zeros_like(tensor)
    for r in range(rank):
        reconstructed_matrix += (
            weights[r] * torch.outer(origin_factors[:, r], destination_factors[:, r])
        )

    # Compute the reconstruction loss (mean squared error)
    loss = torch.norm(tensor - reconstructed_matrix) ** 2

    # Backpropagation
    loss.backward()

    # Update the factors and model parameters
    optimizer.step()

    # Print loss every 100 iterations
    if iteration % 100 == 0:
        print(f"Iteration {iteration}: Loss = {loss.item():.4f}")

# Final results
print("\nFinal Origin Factors (from Covariates):")
print(origin_model(covariates).detach().numpy())  # Nonlinear mapping of covariates
print("\nFinal Destination Factors:")
print(destination_factors.detach().numpy())
print("\nWeights for Each Rank:")
print(weights.detach().numpy())

# Reconstructed matrix
print("\nReconstructed OD Matrix (Approximation):")
print(reconstructed_matrix.detach().numpy())

# Compute final reconstruction error
reconstruction_error = (
    torch.norm(tensor - reconstructed_matrix) / torch.norm(tensor)
).item()
print(f"\nReconstruction Error: {reconstruction_error:.4f}")