import zarr
import numpy as np
import tensorly as tl
from tensorly.decomposition import parafac
import torch


# Set TensorLy backend to PyTorch
tl.set_backend('pytorch')

# Step 1: Load Data from Zarr
zarr_file_path = "data/sample/travel_data_tensor.zarr"  # Replace with your Zarr file path
zarr_store = zarr.open_group(zarr_file_path, mode='r')

# Load the tensor and metadata
travel_data_tensor = zarr_store["travel_data_tensor"][:]
locations = zarr_store["locations"][:].tolist()
weather_conditions = zarr_store["weather_conditions"][:].tolist()
congestion_levels = zarr_store["congestion_levels"][:].tolist()
dates = zarr_store["dates"][:].tolist()
days_of_week = zarr_store["days_of_week"][:].tolist()

print("Travel Data Tensor Shape:", travel_data_tensor.shape)
print("Sample Dimensions:")
print("Locations:", locations[:5])  # Displaying first 5 locations as an example
print("Weather Conditions:", weather_conditions)
print("Congestion Levels:", congestion_levels)
print("Dates:", dates[:5])  # Displaying first 5 dates as an example

# Step 2: Preprocess the Tensor Data
# Replace NaN values with 0 (or you can apply a different imputation method)
travel_data_tensor_cleaned = np.nan_to_num(travel_data_tensor, nan=0)

# Convert the tensor to a PyTorch tensor
travel_data_tensor_pytorch = torch.tensor(travel_data_tensor_cleaned, dtype=torch.float32)

# Define Zarr store path for reconstructed tensors
reconstructed_store_path = "data/sample/reconstructed_tensors.zarr"
reconstructed_store = zarr.open(reconstructed_store_path, mode="w")

# Step 3: Perform CP Decomposition and Store Reconstructed Tensor
rank = 3  # Set the rank for decomposition
weights, factors = parafac(travel_data_tensor_pytorch, rank=rank, init='random')

# Step 4: Reconstruct the Tensor
components = []

for k in range(len(weights)):
    # Start with the weight for the k-th component
    component_k = weights[k]

    # Iteratively compute the outer product across all modes
    for mode, factor in enumerate(factors):
        if mode == 0:  # Initialize with the first factor vector
            component_k = factor[:, k]
        else:
            # Compute the outer product with the next factor manually using PyTorch
            component_k = torch.einsum('i,j->ij', component_k, factor[:, k]).flatten()

    # Reshape the resulting component to match the tensor shape
    component_k = component_k.reshape(travel_data_tensor_pytorch.shape)

    # Store the component
    components.append(component_k)

# Sum all components to reconstruct the full tensor
reconstructed_tensor = torch.stack(components).sum(dim=0)

# Store the reconstructed tensor in Zarr, indexed by rank
reconstructed_store[f"rank_{rank}"] = reconstructed_tensor.cpu().numpy()

# Step 5: Compare with the Original Tensor
difference = torch.norm(reconstructed_tensor - travel_data_tensor_pytorch)
print("Reconstruction Error:", difference.item())

# Confirm the tensor is stored
print(reconstructed_store.tree())



# Open the Zarr store
reconstructed_store = zarr.open("data/sample/reconstructed_tensors.zarr", mode="r")

# Load a tensor for a specific rank
rank_to_load = 3
loaded_tensor = reconstructed_store[f"rank_{rank_to_load}"][:]

print(f"Loaded tensor shape for rank {rank_to_load}: {loaded_tensor.shape}")
