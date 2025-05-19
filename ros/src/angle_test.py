# import numpy as np
# import matplotlib.pyplot as plt
# import numpy as np
# from scipy.optimize import root_scalar

# # Define the function to solve
# def f_forward(x, arr_i):
#     return x - arr_i - (1 / (4 * np.cos((arr_i + x) / 2)))

# # # Example input: arr[i] = -0.25
# # arr_i = -0.25

# # # Solve for arr[i+1] using SciPy's root_scalar
# # solution = root_scalar(f, args=(arr_i,), bracket=[-1, 1], method='brentq')  # Bracket ensures root search

# # arr_next = solution.root
# # print(f"Solved arr[i+1]: {arr_next}")

# # import numpy as np
# # from scipy.optimize import root_scalar

# # Define the function to solve
# def f_backward(x, arr_i):
#     return x - arr_i + (1 / (4 * np.cos((arr_i + x) / 2)))

# # Example input: arr[i] = -0.25
# # arr_i = -0.25

# # # Solve for arr[i-1] using SciPy's root_scalar
# # solution = root_scalar(f, args=(arr_i,), bracket=[-1, 1], method='brentq')  # Bracket ensures root search

# # arr_prev = solution.root
# # print(f"Solved arr[i-1]: {arr_prev}")



# def generate_array(num_elements):
#     if num_elements % 2 != 0:
#         raise ValueError("Number of elements must be even.")

#     a = np.zeros(num_elements)

#     # Set center elements
#     center_idx = num_elements // 2
#     a[center_idx - 1] = -1 / 8
#     a[center_idx] = 1 / 8

#     print(a)
#     # Fill elements outward using the spacing rule
#     for i in range(center_idx - 2, -1, -1):
#         arr_i = a[i+1]
#         soln = root_scalar(f_backward, args=(arr_i,), x0=a[i+2]-a[i+1], method='newton')  # Bracket ensures root search
#         a[i] = soln.root
#         print(a)

#     for i in range(center_idx + 1, num_elements - 1):
#         arr_i = a[i]
#         soln = root_scalar(f_forward, args=(arr_i,), x0=a[i]-a[i-1], method='newton')  # Bracket ensures root search
#         a[i + 1] = soln.root
#         print(a)

#     return a

# # Generate array
# num_elements = 10  # Ensure this is even
# arr = generate_array(num_elements)

# # Compute distances between adjacent elements
# distances = np.diff(arr)

# # Plot distances
# plt.figure(figsize=(8, 4))
# plt.plot(range(len(distances)), distances, 'o-', label="Distance Between Elements")
# plt.xlabel("Index")
# plt.ylabel("Distance")
# plt.title("Distance Between Adjacent Values in Array")
# plt.legend()
# plt.grid()
# plt.show()



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

import numpy as np

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

avg_between = (theta_deg[:-1] + theta_deg[1:]) / 2

print(np.rad2deg(1/(4*np.cos(np.deg2rad(avg_between)))))

# Plot
plt.figure(figsize=(10, 4))
plt.plot(interp_arr[:-1], np.diff(interp_arr), marker='o')
plt.xlabel("Azimuth Angle (degrees)")
plt.ylabel("Angular Resolution (degrees)")
plt.title("Angular Resolution per FFT Bin")
plt.grid(True)
plt.show()
