import numpy as np
from scipy.optimize import minimize

# =======================
# Simulated Data Creation
# =======================
# Define tensor dimensions and other parameters
O, D, T = 5, 5, 10  # O: number of origins, D: number of destinations, T: time steps
R = 2  # Rank of decomposition (number of factors in CP decomposition)
Q = 2  # Number of weather covariates
days_in_week = 7  # Days of the week (one-hot encoded)

# Simulate travel time tensors (real-world data would replace this)
np.random.seed(42)  # Set seed for reproducibility
Tensors = [np.random.rand(O, D) + t * 0.1 for t in range(T)]  # Travel time tensor at each time step

# Simulate weather covariates (e.g., temperature, precipitation) for each time step
weather = np.random.rand(T, Q)  # Shape: (T, Q)

# Simulate day-of-the-week covariates as one-hot encoded vectors
# Example: Monday = [1, 0, 0, 0, 0, 0, 0], Tuesday = [0, 1, 0, 0, 0, 0, 0], etc.
days = np.eye(days_in_week)[np.random.choice(days_in_week, T)]  # Randomly assign days for T time steps

# Combine weather and day-of-the-week into a single covariate matrix
covariates = np.hstack([weather, days])  # Shape: (T, Q + days_in_week)

# ===========================
# Initialize Model Parameters
# ===========================
# Randomly initialize CP decomposition factors
U = np.random.rand(O, R)  # Origin factors, size: (O, R)
V = np.random.rand(D, R)  # Destination factors, size: (D, R)

# Initialize regression coefficients for covariates
Beta_weather = np.random.rand(R, Q)  # Coefficients for weather covariates, size: (R, Q)
Beta_day = np.random.rand(R, days_in_week)  # Coefficients for day-of-the-week covariates, size: (R, days_in_week)
Intercepts = np.random.rand(R)  # Intercept terms for each factor, size: (R,)

# ======================================================
# Helper Function: Compute Temporal Coefficients g(W, d)
# ======================================================
def compute_temporal_coefficients(covariates, Beta_weather, Beta_day, Intercepts):
    """
    Compute temporal coefficients g(W, d) for each time step.

    Parameters:
    - covariates: Combined weather and day-of-the-week data (T, Q + days_in_week)
    - Beta_weather: Coefficients for weather covariates (R, Q)
    - Beta_day: Coefficients for day-of-the-week covariates (R, days_in_week)
    - Intercepts: Intercept terms for each factor (R,)

    Returns:
    - Temporal coefficients: (T, R), where each entry corresponds to the temporal coefficient for factor r at time t.
    """
    weather_terms = covariates[:, :Q] @ Beta_weather.T  # Weather contribution, shape: (T, R)
    day_terms = covariates[:, Q:] @ Beta_day.T  # Day contribution, shape: (T, R)
    return weather_terms + day_terms + Intercepts  # Add contributions and intercept

# ====================================
# Loss Function for Model Optimization
# ====================================
def loss_function(params, Tensors, covariates, O, D, R, Q, days_in_week):
    """
    Compute the loss function for tensor regression.

    Parameters:
    - params: Flattened parameter array (includes U, V, Beta_weather, Beta_day, Intercepts)
    - Tensors: List of observed travel time tensors (one per time step)
    - covariates: Combined covariates (T, Q + days_in_week)
    - O, D: Number of origins and destinations
    - R: Rank of decomposition
    - Q: Number of weather covariates
    - days_in_week: Number of days in the week (7)

    Returns:
    - Total loss (Frobenius norm between observed and reconstructed tensors)
    """
    # Unpack parameters from the flat array
    U = params[:O * R].reshape(O, R)  # Origin factors
    V = params[O * R:O * R + D * R].reshape(D, R)  # Destination factors
    Beta_weather = params[O * R + D * R:O * R + D * R + R * Q].reshape(R, Q)  # Weather coefficients
    Beta_day = params[O * R + D * R + R * Q:O * R + D * R + R * Q + R * days_in_week].reshape(R, days_in_week)  # Day coefficients
    Intercepts = params[O * R + D * R + R * Q + R * days_in_week:]  # Intercept terms

    # Compute temporal coefficients for all time steps
    temporal_coeffs = compute_temporal_coefficients(covariates, Beta_weather, Beta_day, Intercepts)

    # Compute reconstruction loss (Frobenius norm)
    total_loss = 0
    for t in range(len(Tensors)):
        # Reconstruct tensor for time step t
        reconstructed = np.sum(
            [np.outer(U[:, r], V[:, r]) * temporal_coeffs[t, r] for r in range(R)], axis=0
        )
        # Add the Frobenius norm difference to the total loss
        total_loss += np.linalg.norm(Tensors[t] - reconstructed, ord='fro') ** 2
    return total_loss

# ==================
# Model Optimization
# ==================
# Flatten all parameters into a single array for optimization
params_init = np.hstack([
    U.ravel(), V.ravel(), Beta_weather.ravel(), Beta_day.ravel(), Intercepts.ravel()
])

# Minimize the loss function using L-BFGS-B optimization
result = minimize(
    loss_function,
    params_init,
    args=(Tensors, covariates, O, D, R, Q, days_in_week),
    method='L-BFGS-B',  # Quasi-Newton optimization method
    options={'maxiter': 1000, 'disp': True}  # Set maximum iterations and display progress
)

# Unpack optimized parameters
optimized_params = result.x
U_opt = optimized_params[:O * R].reshape(O, R)
V_opt = optimized_params[O * R:O * R + D * R].reshape(D, R)
Beta_weather_opt = optimized_params[O * R + D * R:O * R + D * R + R * Q].reshape(R, Q)
Beta_day_opt = optimized_params[O * R + D * R + R * Q:O * R + D * R + R * Q + R * days_in_week].reshape(R, days_in_week)
Intercepts_opt = optimized_params[O * R + D * R + R * Q + R * days_in_week:]

# ============================
# Tensor Reconstruction Example
# ============================
def reconstruct_tensor(t, U, V, Beta_weather, Beta_day, Intercepts, covariates):
    """
    Reconstruct the travel time tensor for a specific time step t.

    Parameters:
    - t: Time step index
    - U, V: Optimized CP decomposition factors
    - Beta_weather, Beta_day: Optimized regression coefficients
    - Intercepts: Optimized intercept terms
    - covariates: Combined weather and day-of-the-week data

    Returns:
    - Reconstructed tensor for time step t
    """
    temporal_coeffs = compute_temporal_coefficients(covariates, Beta_weather, Beta_day, Intercepts)
    reconstructed = np.sum(
        [np.outer(U[:, r], V[:, r]) * temporal_coeffs[t, r] for r in range(R)], axis=0
    )
    return reconstructed

# Example: Reconstruct the first tensor
reconstructed_tensor_0 = reconstruct_tensor(0, U_opt, V_opt, Beta_weather_opt, Beta_day_opt, Intercepts_opt, covariates)

# Display results
print("Original Tensor at t=0:")
print(Tensors[0])
print("Reconstructed Tensor at t=0:")
print(reconstructed_tensor_0)