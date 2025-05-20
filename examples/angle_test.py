"""
This script demonstrates our angle binning method for computing the equivalent angles for each FFT bin.
"""

import numpy as np
import matplotlib.pyplot as plt

# Parameters
Na = 8                      # number of antennas
lambda_m = 0.00486           # 77 GHz -> 3.9 mm wavelength
la = lambda_m / 2           # spacing
N_bins = Na

# Bin indices centered around zero
k_centered = np.linspace(-N_bins/2 + 0.5, N_bins/2 - 0.5, N_bins)

# Spatial frequencies
f_k = k_centered / (Na * la)

# Corresponding angles
sin_theta = lambda_m * f_k
valid = np.abs(sin_theta) <= 1
theta_rad = np.full(N_bins, np.nan)
theta_rad[valid] = np.arcsin(sin_theta[valid])
theta_deg = np.rad2deg(theta_rad)
print("Angle for each bin:", theta_deg)
print("Difference in angle between bins", np.diff(theta_deg))

avg_between = (theta_deg[:-1] + theta_deg[1:]) / 2
print("Angular resolution at the midpoint between each bin", np.rad2deg(1/(4*np.cos(np.deg2rad(avg_between)))))

def interpolate_array(arr, num_points=4):
    x = np.arange(len(arr))  # Original indices
    x_interp = np.linspace(0, len(arr) - 1, num=(len(arr) - 1) * (num_points + 1) + 1)  # New interpolated indices
    y_interp = np.interp(x_interp, x, arr)  # Perform linear interpolation
    return y_interp

# Example array
interp_arr = interpolate_array(theta_deg, 6)
print(interp_arr.shape)

print(interp_arr)  # Output: Interpolated values


# Angular resolution per bin
theta_res_rad = np.full(len(interp_arr), np.nan)
theta_res_rad = lambda_m / (Na * la * np.cos(np.deg2rad(interp_arr)))
theta_res_deg = np.rad2deg(theta_res_rad)

# Plot
plt.figure(figsize=(10, 4))
plt.plot(interp_arr[:-1], np.diff(interp_arr), marker='o')
plt.xlabel("Azimuth Angle (degrees)")
plt.ylabel("Angular difference (degrees)")
plt.title("Angular difference per FFT Bin")
plt.grid(True)
plt.show()
