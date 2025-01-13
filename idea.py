import numpy as np
from scipy.optimize import minimize

# Simulated Data
O, D, T = 5, 5, 10  # Origins, Destinations, Time steps
R = 2  # Rank of decomposition
Q = 2  # Number of weather covariates
days_in_week = 7

# Simulated travel time tensors
np.random.seed(42)
Tensors = [np.random.rand(O, D) + t * 0.1 for t in range(T)]  # Travel time tensors

# Simulated covariates
weather = np.random.rand(T, Q)  # Weather covariates for each time step
days = np.eye(days_in_week)[np.random.choice(days_in_week, T)]  # One-hot days

# Combine weather and day-of-the-week covariates
covariates = np.hstack([weather, days])  # Shape: (T, Q + days_in_week)

# Initialize CP decomposition factors
U = np.random.rand(O, R)  # Origin factors
V = np.random.rand(D, R)  # Destination factors
Beta_weather = np.random.rand(R, Q)  # Regression coefficients for weather
Beta_day = np.random.rand(R, days_in_week)  # Regression coefficients for day
Intercepts = np.random.rand(R)  # Intercept terms


# Helper function: Compute temporal coefficients g(W, d)
def compute_temporal_coefficients(covariates, Beta_weather, Beta_day, Intercepts):
    weather_terms = covariates[:, :Q] @ Beta_weather.T
    day_terms = covariates[:, Q:] @ Beta_day.T
    return weather_terms + day_terms + Intercepts


# Loss function for optimization
def loss_function(params, Tensors, covariates, O, D, R, Q, days_in_week):
    # Unpack parameters
    U = params[:O * R].reshape(O, R)
    V = params[O * R:O * R + D * R].reshape(D, R)
    Beta_weather = params[O * R + D * R:O * R + D * R + R * Q].reshape(R, Q)
    Beta_day = params[O * R + D * R + R * Q:O * R + D * R + R * Q + R * days_in_week].reshape(R, days_in_week)
    Intercepts = params[O * R + D * R + R * Q + R * days_in_week:]

    # Compute temporal coefficients
    temporal_coeffs = compute_temporal_coefficients(covariates, Beta_weather, Beta_day, Intercepts)

    # Reconstruct tensors and compute loss
    total_loss = 0
    for t in range(len(Tensors)):
        reconstructed = np.sum(
            [np.outer(U[:, r], V[:, r]) * temporal_coeffs[t, r] for r in range(R)], axis=0
        )
        total_loss += np.linalg.norm(Tensors[t] - reconstructed, ord='fro') ** 2
    return total_loss


# Optimization
params_init = np.hstack([
    U.ravel(), V.ravel(), Beta_weather.ravel(), Beta_day.ravel(), Intercepts.ravel()
])
result = minimize(
    loss_function,
    params_init,
    args=(Tensors, covariates, O, D, R, Q, days_in_week),
    method='L-BFGS-B',
    options={'maxiter': 1000, 'disp': True}
)

# Unpack optimized parameters
optimized_params = result.x
U_opt = optimized_params[:O * R].reshape(O, R)
V_opt = optimized_params[O * R:O * R + D * R].reshape(D, R)
Beta_weather_opt = optimized_params[O * R + D * R:O * R + D * R + R * Q].reshape(R, Q)
Beta_day_opt = optimized_params[O * R + D * R + R * Q:O * R + D * R + R * Q + R * days_in_week].reshape(R, days_in_week)
Intercepts_opt = optimized_params[O * R + D * R + R * Q + R * days_in_week:]

# Reconstruct tensors using optimized parameters
def reconstruct_tensor(t, U, V, Beta_weather, Beta_day, Intercepts, covariates):
    temporal_coeffs = compute_temporal_coefficients(covariates, Beta_weather, Beta_day, Intercepts)
    reconstructed = np.sum(
        [np.outer(U[:, r], V[:, r]) * temporal_coeffs[t, r] for r in range(R)], axis=0
    )
    return reconstructed


# Example: Reconstruct the first tensor
reconstructed_tensor_0 = reconstruct_tensor(0, U_opt, V_opt, Beta_weather_opt, Beta_day_opt, Intercepts_opt, covariates)
print("Original Tensor at t=0:")
print(Tensors[0])
print("Reconstructed Tensor at t=0:")
print(reconstructed_tensor_0)
