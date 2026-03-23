"""
Parameters for 2D Single Target Tracking

Constant Velocity (CV) model:
    State:       x = [px, py, vx, vy]^T   (m=4)
    Observation: y = [px, py]^T            (n=2)

    State transition:
        px(k+1) = px(k) + dt * vx(k)
        py(k+1) = py(k) + dt * vy(k)
        vx(k+1) = vx(k)
        vy(k+1) = vy(k)

    Observation:
        y(k) = [px(k), py(k)]^T
"""

import torch

# State and observation dimensions
m = 4  # state: [px, py, vx, vy]
n = 2  # observation: [px, py]

# Sampling interval
delta_t = 1.0  # seconds

##################################
### Initial state and variance ###
##################################
m1_0 = torch.zeros(m, 1)  # initial state mean

#########################################################
### State evolution matrix F (Constant Velocity model) ###
#########################################################
F = torch.tensor([
    [1, 0, delta_t, 0],
    [0, 1, 0, delta_t],
    [0, 0, 1, 0],
    [0, 0, 0, 1]
]).float()

#############################
### Observation matrix H  ###
#############################
H = torch.tensor([
    [1, 0, 0, 0],
    [0, 1, 0, 0]
]).float()

###############################################
### Process noise Q and observation noise R ###
###############################################
# Process noise: continuous white noise acceleration model
# Q = q^2 * G * G^T where G = [dt^2/2, dt^2/2, dt, dt]^T
Q_structure = torch.tensor([
    [delta_t**4 / 4, 0, delta_t**3 / 2, 0],
    [0, delta_t**4 / 4, 0, delta_t**3 / 2],
    [delta_t**3 / 2, 0, delta_t**2, 0],
    [0, delta_t**3 / 2, 0, delta_t**2]
]).float()

# Observation noise: isotropic position noise
R_structure = torch.eye(n).float()


def generate_outlier_mask(T, outlier_ratio=0.1):
    """Generate a boolean mask indicating outlier timesteps."""
    num_outliers = int(T * outlier_ratio)
    mask = torch.zeros(T, dtype=torch.bool)
    if num_outliers > 0:
        outlier_indices = torch.randperm(T)[:num_outliers]
        mask[outlier_indices] = True
    return mask


def add_outliers_to_observations(y, outlier_ratio=0.1, outlier_amplitude=10.0):
    """
    Add outlier spikes to observation sequences for robustness testing.

    Args:
        y: observations [batch_size, n, T] or [n, T]
        outlier_ratio: fraction of timesteps with outliers
        outlier_amplitude: multiplier for outlier magnitude
    Returns:
        y_corrupted: observations with outliers
        outlier_masks: boolean mask of outlier positions
    """
    if y.dim() == 2:
        # Single sequence [n, T]
        T = y.shape[1]
        mask = generate_outlier_mask(T, outlier_ratio)
        y_corrupted = y.clone()
        if mask.any():
            noise = torch.randn_like(y[:, mask]) * outlier_amplitude
            y_corrupted[:, mask] = y[:, mask] + noise
        return y_corrupted, mask

    elif y.dim() == 3:
        # Batch [batch_size, n, T]
        batch_size, n_obs, T = y.shape
        y_corrupted = y.clone()
        outlier_masks = torch.zeros(batch_size, T, dtype=torch.bool)
        for i in range(batch_size):
            mask = generate_outlier_mask(T, outlier_ratio)
            outlier_masks[i] = mask
            if mask.any():
                noise = torch.randn(n_obs, mask.sum().item()) * outlier_amplitude
                y_corrupted[i, :, mask] = y[i, :, mask] + noise
        return y_corrupted, outlier_masks

    else:
        raise ValueError(f"Expected 2D or 3D tensor, got {y.dim()}D")
